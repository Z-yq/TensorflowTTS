# -*- coding: utf-8 -*-
# Copyright 2020 The MelGAN Authors and Minh Nguyen (@dathudeptrai)
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
"""MelGAN Modules."""

import numpy as np

import tensorflow as tf

from models.weight_norm import WeightNormalization


class MultiGeneratorConfig(object):
    """Initialize MelGAN Generator Config."""

    def __init__(self,
                 input_feature='raw',
                 out_channels=25,
                 kernel_size=7,
                 filters=512,
                 hop_size=0.016,
                 sample_rate=8000,
                 stack_kernel_size=3,
                 stacks=3,
                 window=32,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"alpha": 0.2},
                 padding_type="REFLECT",
                 use_final_activation=True,
                 num_mels=80,
                 initializer_seed=42,
                 use_bias=True,
                 is_weight_norm=False,
                 **kwargs):
        if input_feature=='raw':
            assert out_channels==num_mels,'out_channels must equal num_mels while input_feature is "raw"'
        hop_size=int(sample_rate*hop_size)
        if input_feature != 'raw':
            upsample_scales = self.get_scales(hop_size // out_channels)
        else:

            upsample_scales = [1] * 3
        self.is_weight_norm=is_weight_norm
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_mels=num_mels
        self.filters = filters
        self.use_bias=use_bias
        self.window=window
        self.upsample_scales = upsample_scales
        self.stack_kernel_size = stack_kernel_size
        self.stacks = stacks
        self.nonlinear_activation = nonlinear_activation
        self.nonlinear_activation_params = nonlinear_activation_params
        self.padding_type = padding_type
        self.use_final_activation = use_final_activation
        self.initializer_seed = initializer_seed
    def get_scales(self, num):
        scale = []
        while 1:
            for i in range(2, 100):
                if num % i == 0:
                    num = num // i
                    scale.append(i)
                    break
            if num == 1:
                break
        if len(scale)<3:
            scale+=[1]*(3-len(scale))
        scale.sort()
        return scale[::-1]

class MelGANGeneratorConfig(object):
    """Initialize MelGAN Generator Config."""

    def __init__(self,
                 input_feature='raw',
                 num_mels=80,
                 out_channels=80,
                 kernel_size=7,
                 filters=1024,
                 use_bias=True,
                 hop_size=0.016,
                 sample_rate=8000,
                 stack_kernel_size=3,
                 stacks=5,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"alpha": 0.2},
                 padding_type="REFLECT",
                 use_final_nolinear_activation=True,
                 is_weight_norm=False,
                 initializer_seed=42,
                 **kwargs):
        """Init parameters for MelGAN Generator model."""
        if input_feature=='raw':
            assert out_channels==num_mels,'out_channels must equal num_mels while input_feature is "raw"'
        hop_size = int(sample_rate * hop_size)
        if input_feature !='raw':
            upsample_scales =self.get_scales(hop_size//out_channels)
        else:

            upsample_scales =[1]*3
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_mels=num_mels
        self.filters = filters
        self.use_bias = use_bias
        self.upsample_scales = upsample_scales
        self.stack_kernel_size = stack_kernel_size
        self.stacks = stacks
        self.nonlinear_activation = nonlinear_activation
        self.nonlinear_activation_params = nonlinear_activation_params
        self.padding_type = padding_type
        self.use_final_nolinear_activation = use_final_nolinear_activation
        self.is_weight_norm = is_weight_norm
        self.initializer_seed = initializer_seed
    def get_scales(self, num):
        scale = []
        while 1:
            for i in range(2, 100):
                if num % i == 0:
                    num = num // i
                    scale.append(i)
                    break
            if num == 1:
                break
        if len(scale)<3:
            scale+=[1]*(3-len(scale))
        scale.sort()
        return scale[::-1]

class MelGANDiscriminatorConfig(object):
    """Initialize MelGAN Discriminator Config."""

    def __init__(self,
                 dis_out_channels=1,
                 dis_scales=3,
                 dis_downsample_pooling='AveragePooling1D',
                 dis_downsample_pooling_params={
                     "pool_size": 4,
                     "strides": 2,
                 },
                 dis_kernel_sizes=[5, 3],
                 dis_filters=32,
                 dis_max_downsample_filters=1024,
                 dis_use_bias=True,
                 dis_downsample_scales=[2,2, 2,2],
                 dis_nonlinear_activation="LeakyReLU",
                 dis_nonlinear_activation_params={"alpha": 0.2},
                 dis_padding_type="REFLECT",
                 dis_is_weight_norm=True,
                 dis_initializer_seed=42,
                 **kwargs):
        """Init parameters for MelGAN Discriminator model."""
        self.out_channels = dis_out_channels
        self.scales = dis_scales
        self.downsample_pooling = dis_downsample_pooling
        self.downsample_pooling_params = dis_downsample_pooling_params
        self.kernel_sizes = dis_kernel_sizes
        self.filters = dis_filters
        self.max_downsample_filters = dis_max_downsample_filters
        self.use_bias = dis_use_bias
        self.downsample_scales = dis_downsample_scales
        self.nonlinear_activation = dis_nonlinear_activation
        self.nonlinear_activation_params = dis_nonlinear_activation_params
        self.padding_type = dis_padding_type
        self.is_weight_norm = dis_is_weight_norm
        self.initializer_seed = dis_initializer_seed





def get_initializer(initializer_seed=42.):
    """Creates a `tf.initializers.glorot_normal` with the given seed.
    Args:
        initializer_seed: int, initializer seed.
    Returns:
        GlorotNormal initializer with seed = `initializer_seed`.
    """
    return tf.keras.initializers.GlorotNormal(seed=initializer_seed)
class TFMultiWindowGenerator(tf.keras.Model):
    def __init__(self, config, **kwargs):
        """Initialize TFMelGANGenerator module.
        Args:
            config: config object of Melgan generator.
        """
        super(TFMultiWindowGenerator,self).__init__(**kwargs)

        # check hyper parameter is valid or not
        assert config.filters >= np.prod(config.upsample_scales)
        assert config.filters % (2 ** len(config.upsample_scales)) == 0
        self.config = config
        layers = []
        layers += [
            TFReflectionPad1d((config.kernel_size - 1) // 2,
                              padding_type=config.padding_type,
                              name='first_reflect_padding'),
            tf.keras.layers.Conv1D(filters=config.filters,
                                   kernel_size=config.kernel_size,
                                   use_bias=config.use_bias,
                                   kernel_initializer=get_initializer(config.initializer_seed))
        ]

        layers += [
            TFReflectionPad1d((config.kernel_size - 1) // 2,
                              padding_type=config.padding_type,
                              name='second_reflect_padding'),
            tf.keras.layers.Conv1D(filters=config.filters,
                                   kernel_size=config.kernel_size,
                                   use_bias=config.use_bias,
                                   kernel_initializer=get_initializer(config.initializer_seed)),
            TFReflectionPad1d((config.kernel_size - 1) // 2,
                              padding_type=config.padding_type,
                              name='third_reflect_padding'),
            tf.keras.layers.Conv1D(filters=config.filters,
                                   kernel_size=config.kernel_size,
                                   use_bias=config.use_bias,
                                   kernel_initializer=get_initializer(config.initializer_seed))
        ]

        for i, upsample_scale in enumerate(config.upsample_scales):

            # add upsampling layer
            layers += [
                getattr(tf.keras.layers, config.nonlinear_activation)(**config.nonlinear_activation_params),
                TFConvTranspose1d(
                    filters=config.filters // (2 ** (i + 1)),
                    kernel_size=upsample_scale * 2,
                    strides=upsample_scale,
                    padding='same',
                    is_weight_norm=config.is_weight_norm,
                    initializer_seed=config.initializer_seed,
                    name='conv_transpose_._{}'.format(i)
                )

            ]

            # ad residual stack layer
            for j in range(config.stacks):
                layers += [
                    TFResidualStack(
                        kernel_size=config.stack_kernel_size,
                        filters=max(config.filters // (2 ** (i + 1)),128),
                        dilation_rate=config.stack_kernel_size ** j,
                        use_bias=config.use_bias,
                        nonlinear_activation=config.nonlinear_activation,
                        nonlinear_activation_params=config.nonlinear_activation_params,
                        is_weight_norm=config.is_weight_norm,
                        initializer_seed=config.initializer_seed,
                        name='residual_stack_._{}._._{}'.format(i, j)
                    )
                ]
        self.fc=tf.keras.layers.Dense(config.filters)
        self.melgan_feature=tf.keras.Sequential(layers)
        self.reshape1=tf.keras.layers.Reshape([-1, self.config.window * config.filters//2])
        self.reshape2=tf.keras.layers.Reshape([-1, self.config.window * config.filters])
        self.reshape=tf.keras.layers.Reshape([-1,1])

        self.out_layer1= tf.keras.layers.Conv1D(filters=self.config.out_channels, kernel_size=7, strides=1, padding='causal', )
        self.out_layer2 = tf.keras.layers.Dense(self.config.window * self.config.out_channels//2,)
        self.out_layer3 = tf.keras.layers.Dense(self.config.window * self.config.out_channels, )

        self.out_c_score = tf.keras.layers.Conv1D(filters=3, kernel_size=41, padding='causal', activation='softmax',
                                             )




    def _build(self,):
        fake_mels = tf.random.uniform(shape=[1, self.config.window*10, self.config.num_mels], dtype=tf.float32)
        self(fake_mels)
    def call(self,x,training=False,**kwargs):
        features=self.melgan_feature(x,training=training)
        features=self.fc(features)

        out1=self.out_layer1(features)
        out2=self.out_layer2(self.reshape1(features))
        out3=self.out_layer3(self.reshape2(features))
        out1=self.reshape(out1)
        out2=self.reshape(out2)
        out3=self.reshape(out3)
        out_c = tf.keras.layers.concatenate([out1, out2, out3],-1)
        out_c_score=self.out_c_score(out_c,training=training)
        out_wav = tf.reduce_sum(out_c * out_c_score, -1, keepdims=True)
        return out1,out_wav


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

class TFConvUp(tf.keras.layers.Layer):
    """Tensorflow ConvTranspose1d module."""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding,
                 is_weight_norm,
                 initializer_seed,
                 **kwargs):
        """Initialize TFConvTranspose1d( module.
        Args:
            filters (int): Number of filters.
            kernel_size (int): kernel size.
            strides (int): Stride width.
            padding (str): Padding type ("same" or "valid").
        """
        super().__init__(**kwargs)
        self.conv1= tf.keras.layers.Conv1D(
            filters=filters*strides,
            kernel_size=kernel_size,
            # strides=(strides, 1),
            padding="same",
            kernel_initializer=get_initializer(initializer_seed)
        )
        self.up=tf.keras.layers.Reshape([-1,filters])
        self.conv2 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            # strides=(strides, 1),
            padding="same",
            kernel_initializer=get_initializer(initializer_seed)
        )
        if is_weight_norm:
            self.conv1d_transpose = WeightNormalization(self.conv1d_transpose)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T', C').
        """
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)
        return x

class TFConvTranspose1d(tf.keras.layers.Layer):
    """Tensorflow ConvTranspose1d module."""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding,
                 is_weight_norm,
                 initializer_seed,
                 **kwargs):
        """Initialize TFConvTranspose1d( module.
        Args:
            filters (int): Number of filters.
            kernel_size (int): kernel size.
            strides (int): Stride width.
            padding (str): Padding type ("same" or "valid").
        """
        super().__init__(**kwargs)
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding="same",
            kernel_initializer=get_initializer(initializer_seed)
        )
        if is_weight_norm:
            self.conv1d_transpose = WeightNormalization(self.conv1d_transpose)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T', C').
        """
        x = tf.expand_dims(x, 2)
        x = self.conv1d_transpose(x)
        x = tf.squeeze(x, 2)
        return x


class TFResidualStack(tf.keras.layers.Layer):
    """Tensorflow ResidualStack module."""

    def __init__(self,
                 kernel_size,
                 filters,
                 dilation_rate,
                 use_bias,
                 nonlinear_activation,
                 nonlinear_activation_params,
                 is_weight_norm,
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
            getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params),
            TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate),
            tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed)
            ),
            getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params),
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

        # apply weightnorm
        if is_weight_norm:
            self._apply_weightnorm(self.blocks)
            self.shortcut = WeightNormalization(self.shortcut)

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




class TFMelGANGenerator(tf.keras.Model):
    """Tensorflow MelGAN generator module."""

    def __init__(self, config, **kwargs):
        """Initialize TFMelGANGenerator module.
        Args:
            config: config object of Melgan generator.
        """
        super().__init__(**kwargs)

        # check hyper parameter is valid or not
        assert config.filters >= np.prod(config.upsample_scales)
        assert config.filters % (2 ** len(config.upsample_scales)) == 0
        self.config=config
        # add initial layer
        layers = []
        layers += [
            TFReflectionPad1d((config.kernel_size - 1) // 2,
                              padding_type=config.padding_type,
                              name='first_reflect_padding'),
            tf.keras.layers.Conv1D(filters=config.filters,
                                   kernel_size=config.kernel_size,
                                   use_bias=config.use_bias,
                                   kernel_initializer=get_initializer(config.initializer_seed))
        ]



        layers += [
            TFReflectionPad1d((config.kernel_size - 1) // 2,
                              padding_type=config.padding_type,
                              name='second_reflect_padding'),
            tf.keras.layers.Conv1D(filters=config.filters,
                                   kernel_size=config.kernel_size,
                                   use_bias=config.use_bias,
                                   kernel_initializer=get_initializer(config.initializer_seed)),
            TFReflectionPad1d((config.kernel_size - 1) // 2,
                              padding_type=config.padding_type,
                              name='third_reflect_padding'),
            tf.keras.layers.Conv1D(filters=config.filters,
                                   kernel_size=config.kernel_size,
                                   use_bias=config.use_bias,
                                   kernel_initializer=get_initializer(config.initializer_seed))
        ]




        for i, upsample_scale in enumerate(config.upsample_scales):

            # add upsampling layer
            layers += [
                getattr(tf.keras.layers, config.nonlinear_activation)(**config.nonlinear_activation_params),
                TFConvTranspose1d(
                    filters=config.filters // (2 ** (i + 1)),
                    kernel_size=upsample_scale * 2,
                    strides=upsample_scale,
                    padding='same',
                    is_weight_norm=config.is_weight_norm,
                    initializer_seed=config.initializer_seed,
                    name='conv_transpose_._{}'.format(i)
                )

            ]

            # ad residual stack layer
            for j in range(config.stacks):
                layers += [
                    TFResidualStack(
                        kernel_size=config.stack_kernel_size,
                        filters=config.filters // (2 ** (i + 1)),
                        dilation_rate=config.stack_kernel_size ** j,
                        use_bias=config.use_bias,
                        nonlinear_activation=config.nonlinear_activation,
                        nonlinear_activation_params=config.nonlinear_activation_params,
                        is_weight_norm=config.is_weight_norm,
                        initializer_seed=config.initializer_seed,
                        name='residual_stack_._{}._._{}'.format(i, j)
                    )
                ]

        # add final layer

        layers += [
            getattr(tf.keras.layers, config.nonlinear_activation)(**config.nonlinear_activation_params),
            TFReflectionPad1d((config.kernel_size - 1) // 2,
                              padding_type=config.padding_type,
                              name='last_reflect_padding'),
            tf.keras.layers.Conv1D(filters=config.out_channels,
                                   kernel_size=config.kernel_size,
                                   use_bias=config.use_bias,
                                   kernel_initializer=get_initializer(config.initializer_seed))
        ]

        if config.use_final_nolinear_activation:
            layers+= [tf.keras.layers.Activation("tanh")]
        layers+=[tf.keras.layers.Reshape([-1, 1])]
        if config.is_weight_norm is True:
            self._apply_weightnorm(layers)

        self.melgan = tf.keras.models.Sequential(layers)


    # @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 80], dtype=tf.float32)])
    # @tf.function(experimental_relax_shapes=True)
    def call(self, c):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, T, channels)
        Returns:
            Tensor: Output tensor (B, T ** prod(upsample_scales), out_channels)
        """
        x=self.melgan(c)
        return x

    def _build(self):
        """Build model by passing fake input."""
        fake_mels = tf.random.uniform(shape=[1, 200,self.config.num_mels], dtype=tf.float32)
        self(fake_mels)


class TFMelGANDiscriminator(tf.keras.layers.Layer):
    """Tensorflow MelGAN generator module."""

    def __init__(self,
                 config,
                 **kwargs):
        """Initilize MelGAN discriminator module.
        Args:
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15.
                the last two layers' kernel size will be 5 and 3, respectively.
            filters (int): Initial number of filters for conv layer.
            max_downsample_filters (int): Maximum number of filters for downsampling layers.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            padding_type (str): Padding type (support only "REFLECT", "CONSTANT", "SYMMETRIC")
        """
        super().__init__(**kwargs)

        out_channels = config.out_channels
        kernel_sizes = config.kernel_sizes
        filters = config.filters
        max_downsample_filters = config.max_downsample_filters
        use_bias = config.use_bias
        downsample_scales = config.downsample_scales
        nonlinear_activation = config.nonlinear_activation
        nonlinear_activation_params =config.nonlinear_activation_params
        padding_type = config.padding_type
        is_weight_norm = config.is_weight_norm
        initializer_seed = config.initializer_seed
        # check kernel_size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        discriminator = [
            TFReflectionPad1d((np.prod(kernel_sizes) - 1) // 2, padding_type=padding_type),
            tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=int(np.prod(kernel_sizes)),
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed)
            ),
            getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params)
        ]

        # add downsample layers
        in_chs = filters

        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_filters)
            discriminator += [
                tf.keras.layers.Conv1D(out_chs,downsample_scale*10+1,downsample_scale,padding='same',kernel_initializer=get_initializer(initializer_seed))
            ]
            discriminator += [
                getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params)
            ]
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_filters)
        discriminator += [
            tf.keras.layers.Conv1D(
                filters=out_chs,
                kernel_size=kernel_sizes[0],
                padding='same',
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed)
            )
        ]
        discriminator += [
            getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params)
        ]
        discriminator += [
            tf.keras.layers.Conv1D(
                filters=out_channels,
                kernel_size=kernel_sizes[1],
                padding='same',
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed)
            )
        ]

        if is_weight_norm is True:
            self._apply_weightnorm(discriminator)

        self.disciminator = discriminator

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, T, 1).
        Returns:
            List: List of output tensors of each layer.
        """
        outs = []
        for f in self.disciminator:
            x = f(x)
            outs += [x]
        return outs

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass


class TFMelGANMultiScaleDiscriminator(tf.keras.Model):
    """MelGAN multi-scale discriminator module."""

    def __init__(self, config, **kwargs):
        """Initilize MelGAN multi-scale discriminator module.
        Args:
            config: config object for melgan discriminator
        """
        super().__init__(**kwargs)
        self.discriminator = []

        # add discriminator
        for i in range(config.scales):
            self.discriminator += [
                TFMelGANDiscriminator(
                    config,
                    name='melgan_discriminator_scale_._{}'.format(i)
                )
            ]
            self.pooling = getattr(tf.keras.layers, config.downsample_pooling)(**config.downsample_pooling_params)
    def _build(self):
        """Build model by passing fake input."""
        fake_mels = tf.random.uniform(shape=[1, 1600,1], dtype=tf.float32)
        self(fake_mels)
    def call(self, x,**kwargs):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, T, 1).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.discriminator:
            outs += [f(x)]
            x = self.pooling(x)
        return outs
