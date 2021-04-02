# -*- coding: utf-8 -*-
# Copyright 2020 The Tacotron-2 Authors, Minh Nguyen (@dathudeptrai) and Eren Gölge (@erogol)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tacotron-2 Modules."""

import collections

import numpy as np
import tensorflow as tf
from tensorflow_addons.rnn import NASCell, LayerNormLSTMCell, PeepholeLSTMCell
from tensorflow_addons.seq2seq import BahdanauAttention
from tensorflow_addons.seq2seq import Decoder
from tensorflow_addons.seq2seq import Sampler

from models.conformer import ConformerEncoder

from utils.decoder import dynamic_decode


class Tacotron2Config(object):
    """Initialize Tacotron-2 Config."""

    def __init__(
            self,
            vocab_size,
            embedding_hidden_size=512,
            initializer_range=0.02,
            layer_norm_eps=1e-6,
            embedding_dropout_prob=0.1,
            n_speakers=10,
            spk_embed_units=256,

            encoder_type='tacotron',
            n_conv_encoder=5,
            conformer_dmodel=144,
            conformer_fc_factor=0.6,
            conformer_num_heads=4,
            conformer_head_size=32,
            conformer_kernel_size=32,
            conformer_dropout=0.1,

            encoder_conv_filters=512,
            encoder_conv_kernel_sizes=5,
            encoder_conv_activation='mish',
            encoder_conv_dropout_rate=0.1,
            encoder_lstm_units=512,

            outputs_per_step=2,
            n_prenet_layers=2,
            prenet_units=256,
            prenet_activation='mish',
            prenet_dropout_rate=0.1,
            lstm_type='lstm',
            n_lstm_decoder=2,
            decoder_lstm_units=512,
            attention_type='lsa',
            attention_dim=512,
            attention_filters=32,
            attention_kernel=31,
            n_mels=80,
            n_conv_postnet=5,
            postnet_conv_filters=512,
            postnet_conv_kernel_sizes=5,
            postnet_dropout_rate=0.1,
            encoder_out_units=512,
            **kwargs):
        """Init parameters for Tacotron-2 model."""
        self.vocab_size = vocab_size
        self.embedding_hidden_size = embedding_hidden_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_dropout_prob = embedding_dropout_prob
        self.n_speakers = n_speakers
        self.spk_embed_units=spk_embed_units
        # encoder param
        self.encoder_type = encoder_type
        self.n_conv_encoder = n_conv_encoder
        self.conformer_head_size = conformer_head_size
        self.conformer_dmodel = conformer_dmodel
        self.conformer_fc_factor = conformer_fc_factor
        self.conformer_num_heads = conformer_num_heads
        self.conformer_kernel_size = conformer_kernel_size
        self.conformer_dropout = conformer_dropout

        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_sizes = encoder_conv_kernel_sizes
        self.encoder_conv_activation = encoder_conv_activation
        self.encoder_conv_dropout_rate = encoder_conv_dropout_rate
        self.encoder_lstm_units = encoder_lstm_units
        self.encoder_out_units = encoder_out_units
        # decoder param
        self.reduction_factor = outputs_per_step
        self.n_prenet_layers = n_prenet_layers
        self.prenet_units = prenet_units
        self.prenet_activation = prenet_activation
        self.prenet_dropout_rate = prenet_dropout_rate
        self.n_lstm_decoder = n_lstm_decoder
        self.decoder_lstm_units = decoder_lstm_units
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.attention_filters = attention_filters
        self.attention_kernel = attention_kernel
        self.n_mels = n_mels
        self.lstm_type = lstm_type

        # postnet
        self.n_conv_postnet = n_conv_postnet
        self.postnet_conv_filters = postnet_conv_filters
        self.postnet_conv_kernel_sizes = postnet_conv_kernel_sizes
        self.postnet_dropout_rate = postnet_dropout_rate


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
    Args:
        initializer_range: float, initializer range for stddev.
    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def gelu(x):
    """Gaussian Error Linear unit."""
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf


def gelu_new(x):
    """Smoother gaussian Error Linear Unit."""
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    """Swish activation function."""
    return x * tf.sigmoid(x)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


ACT2FN = {
    "identity": tf.keras.layers.Activation('linear'),
    "tanh": tf.keras.layers.Activation('tanh'),
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.layers.Activation(swish),
    "gelu_new": tf.keras.layers.Activation(gelu_new),
    "mish": tf.keras.layers.Activation(mish)
}


class TFReflectionPad1d(tf.keras.layers.Layer):
    """Tensorflow ReflectionPad1d module."""

    def __init__(self, padding_size, padding_type="REFLECT", **kwargs):
        """Initialize TFReflectionPad1d module.

        Args:
            padding_size (int)
            padding_type (str) ("CONSTANT", "REFLECT", or "SYMMETRIC". Default is "REFLECT")
        """
        super().__init__(**kwargs)
        self.padding_size = padding_size
        self.padding_type = padding_type

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Padded tensor (B, T + 2 * padding_size, C).
        """
        return tf.pad(x, [[0, 0], [self.padding_size, self.padding_size], [0, 0]], self.padding_type)


class TFResidualStack(tf.keras.layers.Layer):
    """Tensorflow ResidualStack module."""

    def __init__(self,
                 kernel_size,
                 filters,
                 dilation_rate,
                 use_bias,
                 initializer_seed,
                 **kwargs):
        """Initialize TFResidualStack module.
        Args:
            kernel_size (int): Kernel size.
            filters (int): Number of filters.
            dilation_rate (int): Dilation rate.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
        """
        super().__init__(**kwargs)
        self.blocks = [
            TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate),
            tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed)
            ),

            tf.keras.layers.Conv1D(filters=filters,
                                   kernel_size=1,
                                   use_bias=use_bias,
                                   kernel_initializer=get_initializer(initializer_seed))
        ]
        self.shortcut = tf.keras.layers.Conv1D(filters=filters,
                                               kernel_size=1,
                                               use_bias=use_bias,
                                               kernel_initializer=get_initializer(initializer_seed),
                                               name='shortcut')

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T, C).
        """
        _x = tf.identity(x)
        for layer in self.blocks:
            _x = layer(_x)
        shortcut = self.shortcut(x)
        return shortcut + _x


class TFTacotronConvBatchNorm(tf.keras.layers.Layer):
    """Tacotron-2 Convolutional Batchnorm module."""

    def __init__(self, filters, kernel_size, dropout_rate, activation=None, name_idx=None):
        super().__init__()
        self.conv1d = TFResidualStack(kernel_size, filters, 1, True, 0.02, name='conv_._{}'.format(name_idx))
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, name='ln_norm_._{}'.format(name_idx))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout_._{}'.format(name_idx))
        self.act = ACT2FN[activation]

    def call(self, inputs, training=False):
        outputs = self.conv1d(inputs)
        outputs = self.norm(outputs, training=training)
        outputs = self.act(outputs)
        outputs = self.dropout(outputs, training=training)
        return outputs


class TFTacotronEmbeddings(tf.keras.layers.Layer):
    """Construct character/phoneme/positional/speaker embeddings."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.embedding_hidden_size = config.embedding_hidden_size
        self.initializer_range = config.initializer_range
        self.config = config

        self.speaker_embeddings = tf.keras.layers.Embedding(
            config.n_speakers,
            config.spk_embed_units,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="speaker_embeddings"
        )
        self.speaker_fc = tf.keras.layers.Dense(units=config.embedding_hidden_size, name='speaker_fc')

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.embedding_dropout_prob)

    def build(self, input_shape):
        """Build shared character/phoneme embedding layers."""
        with tf.name_scope("character_embeddings"):
            self.character_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.embedding_hidden_size],
                initializer=get_initializer(self.initializer_range),
            )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Get character embeddings of inputs.
        Args:
            1. character, Tensor (int32) shape [batch_size, length].
            2. speaker_id, Tensor (int32) shape [batch_size]
        Returns:
            Tensor (float32) shape [batch_size, length, embedding_size].
        """
        return self._embedding(inputs, training=training)

    def _embedding(self, inputs, training=False):
        """Applies embedding based on inputs tensor."""
        input_ids, speaker_ids = inputs

        # create embeddings

        inputs_embeds = tf.gather(self.character_embeddings, input_ids)
        embeddings = inputs_embeds
        # print(embeddings.shape)

        speaker_embeddings = self.speaker_embeddings(speaker_ids)
        speaker_embeddings=tf.reduce_mean(speaker_embeddings,1,keepdims=True)
        speaker_features = tf.math.softplus(self.speaker_fc(speaker_embeddings))

        # apply layer-norm and dropout for embeddings.
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings, speaker_features


class TFTacotronEncoderConvs(tf.keras.layers.Layer):
    """Tacotron-2 Encoder Convolutional Batchnorm module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_batch_norm = []
        for i in range(config.n_conv_encoder):
            conv = TFTacotronConvBatchNorm(
                filters=config.encoder_conv_filters,
                kernel_size=config.encoder_conv_kernel_sizes,
                activation=config.encoder_conv_activation,
                dropout_rate=config.encoder_conv_dropout_rate,
                name_idx=i)
            self.conv_batch_norm.append(conv)

    def call(self, inputs, training=False):
        """Call logic."""
        outputs = inputs
        for conv in self.conv_batch_norm:
            outputs = conv(outputs, training=training)
        return outputs


class TFTacotronEncoder(tf.keras.layers.Layer):
    """Tacotron-2 Encoder."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.embeddings = TFTacotronEmbeddings(config, name='embeddings')
        if config.encoder_type == 'conformer':
            self.encoder = ConformerEncoder(config,name='encoder_conformer_part')
        else:

            self.convbn = TFTacotronEncoderConvs(config, name='conv_batch_norm')
            self.bilstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=config.encoder_lstm_units, return_sequences=True),
                name='bilstm'
            )
        self.config=config
        self.fc=tf.keras.layers.Dense(config.encoder_out_units)
    def call(self, inputs, training=False):
        """Call logic."""
        input_ids, speaker_ids, input_mask = inputs

        input_embeddings, speaker_embedding = self.embeddings([input_ids, speaker_ids], training=training)
        # print(input_embeddings.shape,speaker_embedding.shape)
        if self.config.encoder_type=='conformer':
            msha_mask=self.encoder.creat_masks(input_ids)
            outputs = self.encoder(input_embeddings, training=training,mask=msha_mask)
            input_mask = tf.cast(input_mask, tf.float32)
            outputs *= tf.expand_dims(input_mask, -1)
        else:
            outputs=self.convbn(input_embeddings,training=training)
            if len(tf.shape(input_mask)) == 3:
                input_mask = tf.squeeze(input_mask, 1)
            outputs=self.bilstm(outputs,mask=input_mask)
        # print(outputs.shape)
        outputs=self.fc(outputs,training=training)


        return outputs, speaker_embedding


class TrainingSampler(Sampler):
    """Training sampler for Seq2Seq training."""

    def __init__(self,
                 config,
                 ):
        super().__init__()
        self.config = config
        # create schedule factor.
        # the input of a next decoder cell is calculated by formular:
        # next_inputs = ratio * prev_groundtruth_outputs + (1.0 - ratio) * prev_predicted_outputs.
        self._ratio = tf.constant(1.0, dtype=tf.float32)
        self._reduction_factor = self.config.reduction_factor

    def setup_target(self, targets, mel_lengths):
        """Setup ground-truth mel outputs for decoder."""
        self.mel_lengths = mel_lengths
        self.set_batch_size(tf.shape(targets)[0])
        # self.targets = targets[:, self._reduction_factor - 1::self._reduction_factor, :]

        self.targets = tf.reshape(targets, [self._batch_size, -1, self._reduction_factor * self.config.n_mels])
        self.max_lengths = tf.tile([tf.shape(self.targets)[1]], [self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    @property
    def reduction_factor(self):
        return self._reduction_factor

    def initialize(self, ):
        """Return (Finished, next_inputs)."""
        return (tf.tile([False], [self._batch_size]),
                tf.tile([[0.0]], [self._batch_size, self._reduction_factor * self.config.n_mels]))

    def sample(self, time, outputs, state):
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, **kwargs):
        finished = (time + 1 >= self.max_lengths)
        # next_inputs = self._ratio * self.targets[:, time, :] + \
        #     (1.0 - self._ratio) * outputs[:, -self.config.n_mels:]
        next_inputs = tf.cond(tf.random.uniform([1]) <= self._ratio, lambda: self.targets[:, time, :],
                              lambda: outputs[:, -self._reduction_factor * self.config.n_mels:])
        next_state = state
        return (finished, next_inputs, next_state)

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size


class TestingSampler(TrainingSampler):
    """Testing sampler for Seq2Seq training."""

    def __init__(self,
                 config,
                 ):
        super().__init__(config)

    def next_inputs(self, time, outputs, state, sample_ids, **kwargs):
        stop_token_prediction = kwargs.get("stop_token_prediction")
        stop_token_prediction = tf.nn.sigmoid(stop_token_prediction)
        finished = tf.cast(tf.round(stop_token_prediction), tf.bool)
        finished = tf.reduce_all(finished)
        next_inputs = outputs[:, -self._reduction_factor * self.config.n_mels:]
        next_state = state
        return (finished, next_inputs, next_state)


class TFTacotronLocationSensitiveAttention(BahdanauAttention):
    """Tacotron-2 Location Sensitive Attention module."""

    def __init__(self,
                 config,
                 memory,
                 mask_encoder=True,
                 memory_sequence_length=None,
                 is_cumulate=True):
        """Init variables."""
        memory_length = memory_sequence_length if (mask_encoder is True) else None
        super().__init__(
            units=config.attention_dim,
            memory=memory,
            memory_sequence_length=memory_length,
            probability_fn="softmax",
            name="LocationSensitiveAttention"
        )
        self.location_convolution = tf.keras.layers.Conv1D(
            filters=config.attention_filters,
            kernel_size=config.attention_kernel,
            padding='same',
            use_bias=False,
            name='location_conv'
        )
        self.location_layer = tf.keras.layers.Conv1D(filters=config.attention_dim,
                                                     kernel_size=config.attention_kernel,
                                                     padding='same',
                                                     use_bias=False,
                                                     name='location_layer')

        self.v = tf.keras.layers.Dense(1, use_bias=True, name='scores_attention')
        self.config = config
        self.is_cumulate = is_cumulate
        self.use_window = False

    def setup_window(self, win_front=2, win_back=4):
        self.win_front = tf.constant(win_front, tf.int32)
        self.win_back = tf.constant(win_back, tf.int32)

        self._indices = tf.expand_dims(tf.range(tf.shape(self.keys)[1]), 0)
        self._indices = tf.tile(self._indices, [tf.shape(self.keys)[0], 1])  # [batch_size, max_time]

        self.use_window = True

    def _compute_window_mask(self, max_alignments):
        """Compute window mask for inference.
        Args:
            max_alignments (int): [batch_size]
        """
        expanded_max_alignments = tf.expand_dims(max_alignments, 1)  # [batch_size, 1]
        low = expanded_max_alignments - self.win_front
        high = expanded_max_alignments + self.win_back
        mlow = tf.cast((self._indices < low), tf.float32)
        mhigh = tf.cast((self._indices > high), tf.float32)
        mask = mlow + mhigh
        return mask  # [batch_size, max_length]

    def __call__(self, inputs, training=False):
        query, state, prev_max_alignments = inputs

        processed_query = self.query_layer(query) if self.query_layer else query
        processed_query = tf.expand_dims(processed_query, 1)

        expanded_alignments = tf.expand_dims(state, axis=2)
        f = self.location_convolution(expanded_alignments)
        processed_location_features = self.location_layer(f)
        # print(processed_query.shape,processed_location_features.shape,self.keys.shape)
        energy = self._location_sensitive_score(processed_query,
                                                processed_location_features,
                                                self.keys)

        # mask energy on inference steps.
        if self.use_window is True:
            window_mask = self._compute_window_mask(prev_max_alignments)
            energy = energy + window_mask * -1e20

        alignments = self.probability_fn(energy, state)

        if self.is_cumulate:
            state = alignments + state
        else:
            state = alignments

        expanded_alignments = tf.expand_dims(alignments, 2)

        context = tf.reduce_sum(expanded_alignments * self.values, 1)

        return context, alignments, state

    def _location_sensitive_score(self, W_query, W_fil, W_keys):
        """Calculate location sensitive energy."""
        return tf.squeeze(self.v(tf.nn.tanh(W_keys + W_query + W_fil)), -1)

    def get_initial_state(self, batch_size, size):
        """Get initial alignments."""
        return tf.zeros(shape=[batch_size, size], dtype=tf.float32)

    def get_initial_context(self, batch_size):
        """Get initial attention."""
        return tf.zeros(shape=[batch_size, self.config.encoder_out_units], dtype=tf.float32)


class TFTacotronPrenet(tf.keras.layers.Layer):
    """Tacotron-2 prenet."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.prenet_dense = [
            tf.keras.layers.Dense(units=config.prenet_units,
                                  activation=ACT2FN[config.prenet_activation],
                                  name='dense_._{}'.format(i))
            for i in range(config.n_prenet_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate=config.prenet_dropout_rate, name='dropout')

    def call(self, inputs, training=False):
        """Call logic."""
        outputs = inputs
        for layer in self.prenet_dense:
            outputs = layer(outputs)
            outputs = self.dropout(outputs, training=True)
        return outputs


class TFTacotronPostnet(tf.keras.layers.Layer):
    """Tacotron-2 postnet."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_batch_norm = []
        for i in range(config.n_conv_postnet):
            conv = TFTacotronConvBatchNorm(
                filters=config.postnet_conv_filters,
                kernel_size=config.postnet_conv_kernel_sizes,
                dropout_rate=config.postnet_dropout_rate,
                activation='identity' if i + 1 == config.n_conv_postnet else 'tanh',
                name_idx=i,
            )
            self.conv_batch_norm.append(conv)

    def call(self, inputs, training=False):
        """Call logic."""
        outputs = inputs
        for _, conv in enumerate(self.conv_batch_norm):
            outputs = conv(outputs, training=training)
        return outputs


TFTacotronDecoderCellState = collections.namedtuple(
    'TFTacotronDecoderCellState',
    ['attention_lstm_state',
     'decoder_lstms_state',
     'context',
     'time',
     'state',
     'alignment_history',
     'max_alignments', 'speaker_embedding'])

TFDecoderOutput = collections.namedtuple(
    "TFDecoderOutput", ("mel_output", "token_output", "sample_id"))


class TFTacotronDecoderCell(tf.keras.layers.AbstractRNNCell):
    """Tacotron-2 custom decoder cell."""

    def __init__(self, config, training, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.training = training
        self.prenet = TFTacotronPrenet(config)

        # define lstm cell on decoder.
        if config.lstm_type == 'lstm':
            self.attention_lstm = tf.keras.layers.LSTMCell(units=config.decoder_lstm_units,
                                                           name='attention_lstm_cell')
        elif config.lstm_type == 'peephole':
            self.attention_lstm = PeepholeLSTMCell(units=config.decoder_lstm_units,
                                                   name='attention_lstm_cell')
        elif config.lstm_type == 'lnlstm':
            self.attention_lstm = LayerNormLSTMCell(units=config.decoder_lstm_units,
                                                    name='attention_lstm_cell')
        elif config.lstm_type == 'nascell':
            self.attention_lstm = NASCell(units=config.decoder_lstm_units,
                                          name='attention_lstm_cell')
        elif config.lstm_type == 'gru':
            self.attention_lstm = tf.keras.layers.GRUCell(units=config.decoder_lstm_units,
                                                          name='attention_gru_cell')
        else:
            raise ValueError('lstm type not support')

        lstm_cells = []
        for i in range(config.n_lstm_decoder):
            if config.lstm_type == 'lstm':
                lstm_cell = tf.keras.layers.LSTMCell(units=config.decoder_lstm_units,name='decoder__cell_%d'%i
                                                     )
            elif config.lstm_type == 'peephole':
                lstm_cell = PeepholeLSTMCell(units=config.decoder_lstm_units,name='decoder__cell_%d'%i
                                            )
            elif config.lstm_type == 'lnlstm':
                lstm_cell = LayerNormLSTMCell(units=config.decoder_lstm_units,name='decoder__cell_%d'%i
                                              )
            elif config.lstm_type == 'nascell':
                lstm_cell = NASCell(units=config.decoder_lstm_units,name='decoder__cell_%d'%i
                                    )
            elif config.lstm_type == 'gru':
                lstm_cell = tf.keras.layers.GRUCell(units=config.decoder_lstm_units,name='decoder__cell_%d'%i
                                                    )
            else:
                raise ValueError('lstm type not support')

            lstm_cells.append(lstm_cell)
        self.decoder_lstms = tf.keras.layers.StackedRNNCells(lstm_cells,
                                                             name='decoder_lstms')

        # define attention layer.
        if config.attention_type == 'lsa':
            # create location-sensitive attention.
            self.attention_layer = TFTacotronLocationSensitiveAttention(
                config,
                memory=None,
                mask_encoder=True,
                memory_sequence_length=None,
                is_cumulate=True
            )
        else:
            raise ValueError("Only lsa (location-sensitive attention) is supported")

        # frame, stop projection layer.
        self.frame_projection = tf.keras.layers.Dense(
            units=config.n_mels * config.reduction_factor, name='decoder_frame_projection')
        self.stop_projection = tf.keras.layers.Dense(units=config.reduction_factor, name='decoder_stop_projection')

        self.config = config

    def set_alignment_size(self, alignment_size):
        self.alignment_size = alignment_size

    @property
    def output_size(self):
        """Return output (mel) size."""
        return self.frame_projection.units

    @property
    def state_size(self):
        """Return hidden state size."""
        return TFTacotronDecoderCellState(
            attention_lstm_state=self.attention_lstm.state_size,
            decoder_lstms_state=self.decoder_lstms.state_size,
            time=tf.TensorShape([]),
            attention=self.config.attention_dim,
            state=self.alignment_size,
            alignment_history=(),
            max_alignments=tf.TensorShape([1]), speaker_embedding=tf.TensorShape([self.config.embedding_hidden_size])
        )

    def get_initial_state(self, batch_size, speaker_embedding):
        """Get initial states."""
        initial_attention_lstm_cell_states = self.attention_lstm.get_initial_state(None, batch_size, dtype=tf.float32)
        initial_decoder_lstms_cell_states = self.decoder_lstms.get_initial_state(None, batch_size, dtype=tf.float32)
        initial_context = tf.zeros(shape=[batch_size, self.config.encoder_out_units ], dtype=tf.float32)
        initial_state = self.attention_layer.get_initial_state(batch_size, size=self.alignment_size)
        initial_alignment_history = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        return TFTacotronDecoderCellState(
            attention_lstm_state=initial_attention_lstm_cell_states,
            decoder_lstms_state=initial_decoder_lstms_cell_states,
            time=tf.zeros([], dtype=tf.int32),
            context=initial_context,
            state=initial_state,
            alignment_history=initial_alignment_history,
            max_alignments=tf.zeros([batch_size], dtype=tf.int32),
            speaker_embedding=speaker_embedding
        )

    def call(self, inputs, states):
        """Call logic."""
        decoder_input = inputs

        # 1. apply prenet for decoder_input.
        prenet_out = self.prenet(decoder_input, training=self.training)  # [batch_size, dim]

        # 2. concat prenet_out and prev context vector
        # then use it as input of attention lstm layer.
        attention_lstm_input = tf.concat([prenet_out, states.context, states.speaker_embedding], axis=-1)
        attention_lstm_output, next_attention_lstm_state = self.attention_lstm(
            attention_lstm_input, states.attention_lstm_state)

        # 3. compute context, alignment and cumulative alignment.
        prev_state = states.state
        prev_alignment_history = states.alignment_history
        prev_max_alignments = states.max_alignments
        context, alignments, state = self.attention_layer(
            [attention_lstm_output,
             prev_state,
             prev_max_alignments],
            training=self.training
        )

        # 4. run decoder lstm(s)
        # print(context.shape,attention_lstm_output.shape)
        decoder_lstms_input = tf.concat([context, states.speaker_embedding], axis=-1)
        decoder_lstms_output, next_decoder_lstms_state = self.decoder_lstms(
            decoder_lstms_input,
            states.decoder_lstms_state
        )

        # 5. compute frame feature and stop token.
        projection_inputs = tf.concat([decoder_lstms_output, context], axis=-1)
        # projection_inputs = decoder_lstms_output
        decoder_outputs = self.frame_projection(projection_inputs)

        stop_inputs = tf.concat([decoder_lstms_output, decoder_outputs], axis=-1)
        stop_tokens = self.stop_projection(stop_inputs)

        # 6. save alignment history to visualize.
        alignment_history = prev_alignment_history.write(states.time, alignments)

        # 7. return new states.
        new_states = TFTacotronDecoderCellState(
            attention_lstm_state=next_attention_lstm_state,
            decoder_lstms_state=next_decoder_lstms_state,
            time=states.time + 1,
            context=context,
            state=state,
            alignment_history=alignment_history,
            max_alignments=tf.argmax(alignments, -1, output_type=tf.int32),
            speaker_embedding=states.speaker_embedding
        )

        return (decoder_outputs, stop_tokens), new_states


class TFTacotronDecoder(Decoder):
    """Tacotron-2 Decoder."""

    def __init__(self,
                 decoder_cell,
                 decoder_sampler,
                 output_layer=None):
        """Initial variables."""
        self.cell = decoder_cell
        self.sampler = decoder_sampler
        self.output_layer = output_layer

    def setup_decoder_init_state(self, decoder_init_state):
        self.initial_state = decoder_init_state

    def initialize(self, **kwargs):
        return self.sampler.initialize() + (self.initial_state,)

    @property
    def output_size(self):
        return TFDecoderOutput(
            mel_output=tf.nest.map_structure(
                lambda shape: tf.TensorShape(shape), self.cell.output_size),
            token_output=tf.TensorShape(self.sampler.reduction_factor),
            sample_id=self.sampler.sample_ids_shape
        )

    @property
    def output_dtype(self):
        return TFDecoderOutput(
            tf.float32,
            tf.float32,
            self.sampler.sample_ids_dtype
        )

    @property
    def batch_size(self):
        return self.sampler._batch_size

    def step(self, time, inputs, state, training=False):
        (mel_outputs, stop_tokens), cell_state = self.cell(inputs, state, training=training)
        if self.output_layer is not None:
            mel_outputs = self.output_layer(mel_outputs)
        sample_ids = self.sampler.sample(
            time=time, outputs=mel_outputs, state=cell_state
        )
        (finished, next_inputs, next_state) = self.sampler.next_inputs(
            time=time,
            outputs=mel_outputs,
            state=cell_state,
            sample_ids=sample_ids,
            stop_token_prediction=stop_tokens
        )

        outputs = TFDecoderOutput(mel_outputs, stop_tokens, sample_ids)
        return (outputs, next_state, next_inputs, finished)


class TFTacotron2(tf.keras.Model):
    """Tensorflow tacotron-2 model."""

    def __init__(self, config, training, **kwargs):
        """Initalize tacotron-2 layers."""
        super().__init__(self, **kwargs)
        self.encoder = TFTacotronEncoder(config, name='encoder')

        self.decoder_cell = TFTacotronDecoderCell(config, training=training, name='decoder_cell')
        self.decoder = TFTacotronDecoder(
            self.decoder_cell,
            TrainingSampler(config) if training is True else TestingSampler(config)
        )
        self.postnet = TFTacotronPostnet(config, name='post_net')
        self.post_projection = tf.keras.layers.Dense(units=config.n_mels,
                                                     name='residual_projection')

        self.config = config

    def _build(self):
        input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        input_lengths = np.array([9])
        speaker_ids = np.array([[0]])
        mel_outputs = np.random.normal(size=(1, 40, self.config.n_mels)).astype(np.float32)
        mel_lengths = np.array([40])
        self(input_ids, input_lengths, speaker_ids, mel_outputs, mel_lengths,maximum_iterations=10, training=True)

    # @tf.function(experimental_relax_shapes=True)
    def call(self,
             input_ids,
             input_lengths,
             speaker_ids,
             mel_outputs,
             mel_lengths,
             maximum_iterations=2000,
             use_window_mask=False,
             win_front=2,
             win_back=3,
             training=False):
        """Call logic."""
        # create input-mask based on input_lengths


        input_mask = tf.sequence_mask(input_lengths,
                                      maxlen=tf.reduce_max(input_lengths),
                                     )

        encoder_hidden_states, speaker_embedding = self.encoder([input_ids, speaker_ids, input_mask], training=training)

        batch_size = tf.shape(encoder_hidden_states)[0]
        alignment_size = tf.shape(encoder_hidden_states)[1]

        self.decoder.sampler.setup_target(targets=mel_outputs, mel_lengths=mel_lengths)
        self.decoder.cell.set_alignment_size(alignment_size)
        self.decoder.setup_decoder_init_state(
            self.decoder.cell.get_initial_state(batch_size, speaker_embedding[:, 0, :])
        )
        self.decoder.cell.attention_layer.setup_memory(
            memory=encoder_hidden_states,
            memory_sequence_length=input_lengths  # use for mask attention.
        )
        if use_window_mask:
            self.decoder.cell.attention_layer.setup_window(win_front=win_front, win_back=win_back)

        # run decode step.
        (frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(
            self.decoder,
            maximum_iterations=maximum_iterations
        )

        decoder_output = tf.reshape(frames_prediction, [batch_size, -1, self.config.n_mels])
        stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

        residual = self.postnet(decoder_output, training=training)
        residual_projection = self.post_projection(residual)

        mel_outputs = decoder_output + residual_projection

        alignment_history = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

        return decoder_output, mel_outputs, stop_token_prediction, alignment_history

    @tf.function(experimental_relax_shapes=True)
    def inference(self,
                  input_ids,
                  input_lengths,
                  speaker_ids,
                  use_window_mask=False,
                  win_front=2,
                  win_back=4,
                  maximum_iterations=100):
        """Call logic."""
        # create input-mask based on input_lengths
        input_mask = tf.sequence_mask(input_lengths,
                                      maxlen=tf.reduce_max(input_lengths),
                                     )

        encoder_hidden_states, speaker_embedding = self.encoder([input_ids, speaker_ids, input_mask], training=False)

        batch_size = tf.shape(encoder_hidden_states)[0]
        alignment_size = tf.shape(encoder_hidden_states)[1]

        self.decoder.sampler.set_batch_size(batch_size)
        self.decoder.cell.set_alignment_size(alignment_size)
        self.decoder.setup_decoder_init_state(
            self.decoder.cell.get_initial_state(batch_size, speaker_embedding[:, 0, :])
        )
        # input_lengths = tf.squeeze(input_lengths, -1)
        self.decoder.cell.attention_layer.setup_memory(
            memory=encoder_hidden_states,
            memory_sequence_length=input_lengths  # use for mask attention.
        )
        if use_window_mask:
            self.decoder.cell.attention_layer.setup_window(win_front=win_front, win_back=win_back)

        (frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(
            self.decoder,
            maximum_iterations=maximum_iterations
        )

        decoder_output = tf.reshape(frames_prediction, [batch_size, -1, self.config.n_mels])
        stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

        residual = self.postnet(decoder_output, training=False)
        residual_projection = self.post_projection(residual)

        mel_outputs = decoder_output + residual_projection

        alignment_history = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

        return decoder_output, mel_outputs, stop_token_prediction, alignment_history
