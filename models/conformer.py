
import tensorflow as tf
from models.positional_encoding import PositionalEncoding
from utils.tools import merge_two_last_dims,shape_list
from models.layers.switchnorm import SwitchNormalization
from models.layers.multihead_attention import MultiHeadAttention

class GLU(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        a, b = tf.split(inputs, 2, axis=self.axis)
        b = tf.nn.sigmoid(b)
        return tf.multiply(a, b)

    def get_config(self):
        conf = super(GLU, self).get_config()
        conf.update({"axis": self.axis})
        return conf




class FFModule(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 dropout=0.0,
                 fc_factor=0.5,
                 name="ff_module",
                 **kwargs):
        super(FFModule, self).__init__(name=name, **kwargs)
        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization()
        self.ffn1 = tf.keras.layers.Dense(4 * input_dim)
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name="swish_activation")
        self.do1 = tf.keras.layers.Dropout(dropout)
        self.ffn2 = tf.keras.layers.Dense(input_dim)
        self.do2 = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.res_add([inputs, self.fc_factor * outputs])
        return outputs

    def get_config(self):
        conf = super(FFModule, self).get_config()
        conf.update({"fc_factor": self.fc_factor})
        conf.update(self.ln.get_config())
        conf.update(self.ffn1.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.do1.get_config())
        conf.update(self.ffn2.get_config())
        conf.update(self.do2.get_config())
        conf.update(self.res_add.get_config())
        return conf


class MHSAModule(tf.keras.layers.Layer):
    def __init__(self,
                 head_size,
                 num_heads,
                 dropout=0.0,
                 name="mhsa_module",
                 **kwargs):
        super(MHSAModule, self).__init__(name=name, **kwargs)
        # self.pc = PositionalEncoding()
        self.ln = tf.keras.layers.LayerNormalization()
        self.mha = MultiHeadAttention(head_size=head_size, num_heads=num_heads)
        self.do = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False,mask=None, **kwargs):
        # outputs = self.pc(inputs)
        outputs = self.ln(inputs, training=training)
        outputs = self.mha([outputs, outputs, outputs], training=training,mask=mask)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(MHSAModule, self).get_config()
        # conf.update(self.pc.get_config())
        conf.update(self.ln.get_config())
        conf.update(self.mha.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf
def create_padding_mask(seq):
    seq_pad = tf.cast(tf.equal(seq, 0), tf.float32)
    # seq_pad = tf.clip_by_value(seq_pad,0.,1.)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq_pad[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
def create_masks(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return  combined_mask
class ConvModule(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 kernel_size=32,
                 dropout=0.0,
                 name="conv_module",
                 **kwargs):
        super(ConvModule, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization()
        self.pw_conv_1 = tf.keras.layers.Conv1D(
            filters=2 * input_dim, kernel_size=1, strides=1,
            padding="same", name="pw_conv_1"
        )
        self.glu = GLU()
        self.dw_conv = tf.keras.layers.SeparableConv1D(
            filters=2 * input_dim, kernel_size=kernel_size, strides=1,
            padding="same", depth_multiplier=1, name="dw_conv"
        )
        self.bn =SwitchNormalization()
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name="swish_activation")
        self.pw_conv_2 = tf.keras.layers.Conv1D(filters=input_dim, kernel_size=1, strides=1,
                                                padding="same", name="pw_conv_2")
        self.do = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(ConvModule, self).get_config()
        conf.update(self.ln.get_config())
        conf.update(self.pw_conv_1.get_config())
        conf.update(self.glu.get_config())
        conf.update(self.dw_conv.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.pw_conv_2.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 dropout=0.0,
                 fc_factor=0.5,
                 head_size=144,
                 num_heads=4,
                 kernel_size=32,
                 name="conformer_block",
                 **kwargs):
        super(ConformerBlock, self).__init__(name=name, **kwargs)
        self.ffm1 = FFModule(input_dim=input_dim,
                             dropout=dropout, fc_factor=fc_factor,
                             name="ff_module_1")
        self.mhsam = MHSAModule(head_size=head_size, num_heads=num_heads,
                                dropout=dropout)
        self.convm = ConvModule(input_dim=input_dim, kernel_size=kernel_size,
                                dropout=dropout)
        self.ffm2 = FFModule(input_dim=input_dim,
                             dropout=dropout, fc_factor=fc_factor,
                             name="ff_module_2")
        self.ln = tf.keras.layers.LayerNormalization()

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False,mask=None, **kwargs):
        outputs = self.ffm1(inputs, training=training)
        outputs = self.mhsam(outputs, training=training,mask=mask)
        outputs = self.convm(outputs, training=training)
        outputs = self.ffm2(outputs, training=training)
        outputs = self.ln(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(ConformerBlock, self).get_config()
        conf.update(self.ffm1.get_config())
        conf.update(self.mhsam.get_config())
        conf.update(self.convm.get_config())
        conf.update(self.ffm2.get_config())
        conf.update(self.ln.get_config())
        return conf

class ConformerEncoder(tf.keras.Model):
    def __init__(self,
                 config,
                 name="conformer_encoder",
                 **kwargs):
        super(ConformerEncoder, self).__init__(name=name, **kwargs)
        self.dmodel=config.conformer_dmodel
        self.projector=tf.keras.layers.Dense(config.conformer_dmodel)
        self.dropout=config.conformer_dropout
        self.conformer_blocks = []
        for i in range(config.n_conv_encoder):
            conformer_block = ConformerBlock(
                input_dim=config.conformer_dmodel,
                dropout=config.conformer_dropout,
                fc_factor=config.conformer_fc_factor,
                head_size=config.conformer_head_size,
                num_heads=config.conformer_num_heads,
                kernel_size=config.conformer_kernel_size,
                name=f"conformer_block_{i}"
            )
            self.conformer_blocks.append(conformer_block)
    def creat_masks(self,inp):
        return create_masks(inp)
    def call(self, inputs, training=False,mask=None, **kwargs):
        outputs=self.projector(inputs,training=training)
        for cblock in self.conformer_blocks:
            outputs = cblock(outputs, training=training,mask=mask)
        return outputs

    def get_config(self):
        conf = super(ConformerEncoder, self).get_config()
        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf


