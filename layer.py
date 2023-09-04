import tensorflow as tf
from tensorflow.keras import layers

class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(name='scale', 
                                 shape=input_shape[-1:], 
                                 initializer=tf.random_normal_initializer(1., 0.02), 
                                 trainable=True)

    self.offset = self.add_weight(name='offset', 
                                  shape=input_shape[-1:], 
                                  initializer='zeros', 
                                  trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset


def conv_block(filters, size=3, strides=2, dilation_rate=1, norm_type=None, activation='lrelu', downsample=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    sequential = tf.keras.Sequential()
    if downsample:
        sequential.add(
            tf.keras.layers.Conv2D(filters, size, strides=strides,
                                   padding='same', dilation_rate=dilation_rate, 
                                   kernel_initializer=initializer, use_bias=False))
    else:
        sequential.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, 
                                            padding='same', kernel_initializer=initializer, 
                                            use_bias=False))
        if norm_type is not None:
            if norm_type == 'instance':
                sequential.add(InstanceNormalization())
            elif norm_type == 'batch':
                sequential.add(tf.keras.layers.BatchNormalization())

        if activation == 'lrelu':
            sequential.add(tf.keras.layers.LeakyReLU())
        elif activation == 'relu':
            sequential.add(tf.keras.layers.ReLU())
    return sequential


class SE_Block(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16):
        super(SE_Block, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.initializer = tf.random_normal_initializer(0., 0.02)

    def build(self, input_shape):
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.conv1 = tf.keras.layers.Conv2D(filters=self.channels // self.reduction_ratio, 
                                            kernel_size=1, 
                                            activation='relu', 
                                            kernel_initializer=self.initializer)
        
        self.conv2 = tf.keras.layers.Conv2D(filters=self.channels, 
                                            kernel_size=1, 
                                            activation='sigmoid', 
                                            kernel_initializer=self.initializer)

    def call(self, inputs):
        batch_size = inputs.shape[0]
        pooled_inputs = self.global_pooling(inputs)
        reshaped_inputs = tf.reshape(pooled_inputs, [batch_size, 1, 1,-1])
        conv_output = self.conv1(reshaped_inputs)
        conv_output = self.conv2(conv_output)
        return tf.math.multiply(inputs, conv_output)


class ASPP(tf.keras.layers.Layer):
    def __init__(self, channels, rates=[2, 4, 8, 16]):
        super(ASPP, self).__init__()
        self.channels = channels
        self.rates = rates

    def build(self, input_shape):
        self.conv1x1 =  conv_block(self.channels, size=1, strides=1, 
                                   norm_type='instance', activation='lrelu')

        self.conv3x3_1 = conv_block(self.channels, size=3, strides=1, 
                                    dilation_rate=self.rates[0], 
                                    norm_type='instance', activation='lrelu')
        
        self.conv3x3_2 = conv_block(self.channels, size=3, strides=1, 
                                    dilation_rate=self.rates[1], 
                                    norm_type='instance', activation='lrelu')
        
        self.conv3x3_3 = conv_block(self.channels, size=3, strides=1, 
                                    dilation_rate=self.rates[2], 
                                    norm_type='instance', activation='lrelu')
        
        # self.conv3x3_4 = conv_block(self.channels, size=3, strides=1, dilation_rate=self.rates[3], norm_type='instance', activation='lrelu')

        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.conv1x1_pool = conv_block(self.channels, size=1, strides=1, 
                                       norm_type='instance', activation='lrelu')

        self.concat = tf.keras.layers.Concatenate()
        self.conv1x1_concat = conv_block(self.channels, size=1, strides=1, 
                                         norm_type='instance', activation='lrelu')

    def call(self, inputs):
        batch_size = inputs.shape[0]
        x1 = self.conv1x1(inputs)
        x2 = self.conv3x3_1(inputs)
        x3 = self.conv3x3_2(inputs)
        x4 = self.conv3x3_3(inputs)
        # x5 = self.conv3x3_4(inputs)

        x5 = self.global_pooling(inputs)
        x5 = tf.reshape(x5, [batch_size, 1, 1, -1])
        x5 = self.conv1x1_pool(x5)
        x5 = tf.image.resize(x5, (inputs.shape[1], inputs.shape[2]))

        x = self.concat([x1, x2, x3, x4, x5])
        x = self.conv1x1_concat(x)
        return x
