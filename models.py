import tensorflow as tf
from tensorflow.keras import layers
from layer import *

class Mask_G(tf.keras.Model):
    def __init__(self, filters=32, block_num=5):
        super(Mask_G, self).__init__()
        self.filters = filters
        self.block_num = block_num
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.encoders = []
        self.decoders = []

    def build(self, input_shape):
        # Encoder layers
        for i in range(self.block_num):
            norm_type = 'instance' if i != 0 else None
            downsample_block = conv_block(self.filters, size=3, strides=2, 
                                          norm_type=norm_type, activation='lrelu')
            
            self.encoders.append(downsample_block)

            self.filters = self.filters*2 if i < 4 else self.filters

        self.encoders.append(conv_block(self.filters, size=3, strides=2, 
                                        norm_type='instance', activation='lrelu'))
        # Dencoder layers
        for i in range(self.block_num):
            self.decoders.append(conv_block(self.filters, size=3, strides=2, 
                                            norm_type='instance', activation='relu',  
                                            downsample=False))
            self.filters//=2

        self.out = layers.Conv2DTranspose(1, 
                                          kernel_size=3, 
                                          strides=2, 
                                          padding='same',
                                          kernel_initializer=self.initializer, 
                                          use_bias=False)

    def call(self, inputs, training=False):
        x = inputs
        encoders_outputs = []
        for i, block in enumerate(self.encoders):
            x = block(x)
            if 0 < i < self.block_num:
                encoders_outputs.append(x)
        for i, block in enumerate(self.decoders):
            x = block(x)
            if i < self.block_num-1:
                x = tf.concat([encoders_outputs.pop(), x], -1)
        out = self.out(x)

        return tf.keras.activations.tanh(out)


class Face_G(tf.keras.Model):
    def __init__(self, filters=32, block_num=5):
        super(Face_G, self).__init__()
        self.filters = filters
        self.block_num = block_num
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.encoders = []
        self.decoders = []

    def build(self, input_shape):
        # Encoder layers
        for i in range(self.block_num):
            norm_type = 'instance' if i != 0 else None
            if i < 3:
                downsample_block = [conv_block(self.filters, size=3, strides=2, 
                                                norm_type=norm_type, activation='lrelu'), 
                                    SE_Block(self.filters)]
                downsample_block = tf.keras.Sequential(downsample_block)                    
            else:
                downsample_block = conv_block(self.filters, size=3, strides=2, 
                                                  norm_type=norm_type, activation='lrelu')
                
            self.encoders.append(downsample_block)

            self.filters = self.filters*2 if i < 4 else self.filters

        self.encoders.append(ASPP(self.filters))

        # Dencoder layers
        for i in range(self.block_num):
            strides = 1 if i == 0 else 2
            self.decoders.append(conv_block(self.filters, size=3, strides=strides, 
                                            norm_type='instance', activation='relu',
                                            downsample=False))
            self.filters//=2

        self.out = layers.Conv2DTranspose(3, 
                                          kernel_size=3, 
                                          strides=2, 
                                          padding='same',
                                          kernel_initializer=self.initializer, 
                                          use_bias=False)

    def call(self, inputs, training=False):
        face, mask = inputs[0], inputs[1]
        x = tf.concat([face, mask], -1)
        encoders_outputs = []
        for i, block in enumerate(self.encoders):
            x = block(x)
            if 0 < i < self.block_num: 
                encoders_outputs.append(x)
        for i, block in enumerate(self.decoders):
            x = block(x)
            if i < self.block_num-1:
                x = tf.concat([encoders_outputs.pop(), x], -1)
        out = self.out(x)

        return tf.keras.activations.tanh(out)


class Face_D_whole(tf.keras.Model):
  def __init__(self, filters=64):
    super(Face_D_whole, self).__init__()
    self.filters = filters
    self.initializer = tf.random_normal_initializer(0., 0.02)

  def build(self, input_shape):
      self.down1_3channel = conv_block(self.filters, size=4, strides=2, 
                                       norm_type='batch', activation='lrelu')

      self.down1_6channel = conv_block(self.filters, size=4, strides=2, 
                                       norm_type='batch', activation='lrelu')

      self.down2 = conv_block(self.filters*2, size=4, strides=2, 
                              norm_type='batch', activation='lrelu')

      self.down3 = conv_block(self.filters*4, size=4, strides=2, 
                              norm_type='batch', activation='lrelu')

      self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
      self.conv = tf.keras.layers.Conv2D(self.filters*8, 
                                         4, 
                                         strides=1,
                                         kernel_initializer=self.initializer,
                                         use_bias=False)

      self.batchnorm = tf.keras.layers.BatchNormalization()
      self.leaky_relu = tf.keras.layers.LeakyReLU()
      self.zero_pad2 = tf.keras.layers.ZeroPadding2D()

      self.last = tf.keras.layers.Conv2D(1, 
                                         4, 
                                         strides=1, 
                                         kernel_initializer=self.initializer)

  def call(self, inputs, training=False):
      if isinstance(inputs, list):
          gen_img, target = inputs[0], inputs[1]
          x = tf.concat([gen_img, target], -1)
          x = self.down1_6channel(x)
      else:
          x = inputs
          x = self.down1_3channel(x)

      x = self.down2(x)
      x = self.down3(x)

      x = self.zero_pad1(x)
      x = self.conv(x)
      x = self.batchnorm(x, training=training)
      x = self.leaky_relu(x)
      x = self.zero_pad2(x)

      return self.last(x)


class Face_D_region(tf.keras.Model):
  def __init__(self, filters=64):
    super(Face_D_region, self).__init__()
    self.filters = filters
    self.initializer = tf.random_normal_initializer(0., 0.02)

  def build(self, input_shape):
      self.down1_3channel = conv_block(self.filters, size=4, strides=2, 
                                       norm_type='batch', activation='lrelu')

      self.down1_6channel = conv_block(self.filters, size=4, strides=2, 
                                       norm_type='batch', activation='lrelu')

      self.down2 = conv_block(self.filters*2, size=4, strides=2, 
                              norm_type='batch', activation='lrelu')

      self.down3 = conv_block(self.filters*4, size=4, strides=2, 
                              norm_type='batch', activation='lrelu')

      self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
      self.conv = tf.keras.layers.Conv2D(self.filters*8, 
                                         4, 
                                         strides=1, 
                                         kernel_initializer=self.initializer, 
                                         use_bias=False)

      self.batchnorm = tf.keras.layers.BatchNormalization()
      self.leaky_relu = tf.keras.layers.LeakyReLU()
      self.zero_pad2 = tf.keras.layers.ZeroPadding2D()

      self.last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=self.initializer)

  def call(self, inputs, training=False):
      if isinstance(inputs, list):
          input_img, mask_map, gen_img, target = inputs[0], inputs[1], inputs[2], inputs[3]
          mask_region = tf.where(mask_map == 1, gen_img, input_img)
          x = tf.concat([mask_region, target], -1)
          x = self.down1_6channel(x)
      else:
          x = inputs
          x = self.down1_3channel(x)

      x = self.down2(x)
      x = self.down3(x)

      x = self.zero_pad1(x)
      x = self.conv(x)
      x = self.batchnorm(x, training=training)
      x = self.leaky_relu(x)
      x = self.zero_pad2(x)

      return self.last(x)


class VGG19_model():
  def __init__(self):
    selected_layers = ["block3_conv4", "block4_conv4", "block5_conv4"]
    self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    self.vgg.trainable = False
    self.outputs = [self.vgg.get_layer(l).output for l in selected_layers]

  def get_vgg19(self):
    vgg_model = tf.keras.Model(self.vgg.input, self.outputs)
    return vgg_model
