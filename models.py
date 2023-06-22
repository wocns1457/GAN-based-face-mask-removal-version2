import tensorflow as tf
from tensorflow.keras import layers
from layer import BatchInstanceNormalization

class Mask_G(tf.keras.Model):
  def __init__(self, filters=64, kernel_size=3, strides=1, padding='same'):
    super(Mask_G, self).__init__()
    initializer = tf.random_normal_initializer(0., 0.02)
    self.strides = strides
    self.kernel_size = kernel_size
    self.padding = padding

    self.encoders = []
    self.decoders = []

    # Encoder layers
    self.start_conv = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.pool = layers.MaxPooling2D(pool_size=(2, 2)) 
    
    for i in range(0, 5):
        conv1 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
        conv2 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
        bn1 = BatchInstanceNormalization() 
        bn2 = BatchInstanceNormalization() 
        
        if i == 0:
            encoder_block = [conv1, bn1]
        else:
            encoder_block = [conv1, bn1, conv2, bn2]
        self.encoders.append(encoder_block)
  
        filters = filters*2 if i < 3 else filters
    
    # Dencoder layers
    for i in range(0, 4):
        conv1 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
        conv2 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
        convt = layers.Conv2DTranspose(filters, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)
        bn1 = BatchInstanceNormalization() 
        bn2 = BatchInstanceNormalization() 

        decoder_block = [convt, conv1, bn1, conv2, bn2]

        self.decoders.append(decoder_block)

        filters//=2
    self.out = layers.Conv2D(1, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)

  def call(self, inputs, training=False):
    encoders_outputs = []
    
    enc = self.start_conv(inputs)

    for i, block in enumerate(self.encoders):
        for layer in block:
            if isinstance(layer, BatchInstanceNormalization):
                enc = layer(enc, training=training)
                enc = tf.nn.leaky_relu(enc)
            else:
                enc = layer(enc)
        if i < 4:
            encoders_outputs.append(enc)
            enc = self.pool(enc)
    brige = enc
    dec = brige

    for i, block in enumerate(self.decoders):
        for layer in block:
            if isinstance(layer, layers.Conv2DTranspose):
                dec = layer(dec)
                dec = layers.Concatenate()([encoders_outputs.pop(), dec])
            elif isinstance(layer, BatchInstanceNormalization):
                dec = layer(dec, training=training)
                dec = tf.nn.leaky_relu(dec)
            else:
                dec = layer(dec)
    out = self.out(dec)
    return tf.keras.activations.tanh(out)

class Face_G(tf.keras.Model):
  def __init__(self, filters=64, kernel_size=3, strides=1, padding='same'):
    super(Face_G, self).__init__()
    initializer = tf.random_normal_initializer(0., 0.02)
    self.strides = strides
    self.kernel_size = kernel_size
    self.padding = padding

    self.encoders = []
    self.decoders = []

    # Encoder layers
    self.start_conv1 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.start_conv2 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.pool = layers.MaxPooling2D(pool_size=(2, 2)) 
    
    dilation_rate = [1, 1, 2, 4, 8, 16]
    for i in range(0, 3):
        filters = filters*2 if i < 2 else filters
        dilation1, dilation2 = dilation_rate[i*2 : i*2+2][0], dilation_rate[i*2 : i*2+2][1]
        conv1 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, dilation_rate=dilation1, use_bias=False)
        conv2 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, dilation_rate=dilation2, use_bias=False)
        bn1 = BatchInstanceNormalization() 
        bn2 = BatchInstanceNormalization() 
        
        encoder_block = [conv1, bn1, conv2, bn2]

        if i < 2:
            self.encoders.append(encoder_block)
        else:
            self.encoders[-1].extend(encoder_block)
        
    # Dencoder layers
    for i in range(0, 2):
        filters//=2
        conv1 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
        conv2 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
        convt = layers.Conv2DTranspose(filters, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)
        bn1 = BatchInstanceNormalization() 
        bn2 = BatchInstanceNormalization() 

        decoder_block = [convt, conv1, bn1, conv2, bn2]

        self.decoders.append(decoder_block)

    self.last_conv = layers.Conv2D(3, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.last_bn = BatchInstanceNormalization() 
    self.out = layers.Conv2D(3, kernel_size=1, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)

  def call(self, inputs, training=False):
    input_mask = inputs[0]
    input_map = inputs[1]
    inputs = layers.concatenate([input_mask, input_map])

    encoders_outputs = []
    
    enc = self.start_conv1(inputs)
    enc = self.start_conv2(enc)
    enc = tf.nn.leaky_relu(enc)
    encoders_outputs.append(enc)
    enc = self.pool(enc)

    for i, block in enumerate(self.encoders):
        for layer in block:
            if isinstance(layer, BatchInstanceNormalization):
                enc = layer(enc, training=training)
                enc = tf.nn.leaky_relu(enc)
            else:
                enc = layer(enc)
        if i < 1:
            encoders_outputs.append(enc)
            enc = self.pool(enc)
    brige = enc
    dec = brige
    
    for i, block in enumerate(self.decoders):
        for layer in block:
            if isinstance(layer, layers.Conv2DTranspose):
                dec = layer(dec)
                dec = layers.Concatenate()([encoders_outputs.pop(), dec])
            elif isinstance(layer, BatchInstanceNormalization):
                dec = layer(dec, training=training)
                dec = tf.nn.leaky_relu(dec)
            else:
                dec = layer(dec)
    
    dec = self.last_conv(dec)
    dec = self.last_bn(dec)
    dec = tf.nn.leaky_relu(dec)

    out = self.out(dec)
    return tf.keras.activations.tanh(out)

class Face_D_whole(tf.keras.Model):
    def __init__(self, filters=64, kernel_size=4, strides=1, padding='same'):
        super(Face_D_whole, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.strides = strides
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1_1 = layers.Conv2D(filters*1, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)

        self.conv2_1 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)
        self.bn2_1 = BatchInstanceNormalization()

        self.conv3_1 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)
        self.bn3_1 = BatchInstanceNormalization()

        self.zero_pad1 = layers.ZeroPadding2D()
        self.conv4_1 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding='valid', kernel_initializer=initializer, use_bias=False)
        self.bn4_1 = BatchInstanceNormalization()

        self.zero_pad2 = layers.ZeroPadding2D()
        self.conv5_1 = layers.Conv2D(1, kernel_size=self.kernel_size, strides=self.strides, padding='valid', kernel_initializer=initializer)

    def call(self, inputs, training=False):  
        x = self.conv1_1(inputs)
        x = tf.nn.leaky_relu(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = tf.nn.leaky_relu(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad1(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = tf.nn.leaky_relu(x)
        
        x = self.zero_pad2(x)
        x = self.conv5_1(x)
        return x

class Face_D_region(tf.keras.Model):
    def __init__(self, filters=64, kernel_size=4, strides=1, padding='same'):
        super(Face_D_region, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.strides = strides
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1_1 = layers.Conv2D(filters*1, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)

        self.conv2_1 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)
        self.bn2_1 = BatchInstanceNormalization()

        self.conv3_1 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)
        self.bn3_1 = BatchInstanceNormalization()

        self.zero_pad1 = layers.ZeroPadding2D()
        self.conv4_1 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding='valid', kernel_initializer=initializer, use_bias=False)
        self.bn4_1 = BatchInstanceNormalization()

        self.zero_pad2 = layers.ZeroPadding2D()
        self.conv5_1 = layers.Conv2D(1, kernel_size=self.kernel_size, strides=self.strides, padding='valid', kernel_initializer=initializer)

    def prepare_input_disc_mask(self, x):
        Igt_Iedit = x[0]
        Imask_map = x[1]
        Iinput = x[2]
        Imask_map = Imask_map/255.0
        complementary = 1-Imask_map
        firstmul = layers.Multiply()([Iinput, complementary])
        secondmul = layers.Multiply()([Igt_Iedit, Imask_map])
        Imask_region = layers.Add()([firstmul, secondmul])
        return Imask_region

    def call(self, inputs, training=False):  #input.shape : ([None, 128, 128, 3], [None, 128, 128, 1], [None, 128, 128, 3])
        Igt_Iedit = inputs[0] # Ground truth or generated image
        Imask_map = inputs[1] # Mask map image
        Iinput = inputs[2] # Original image
        input = layers.Lambda(self.prepare_input_disc_mask)([Igt_Iedit, Imask_map, Iinput]) # preparation input

        x = self.conv1_1(input)
        x = tf.nn.leaky_relu(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = tf.nn.leaky_relu(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad1(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = tf.nn.leaky_relu(x)
        
        x = self.zero_pad2(x)
        x = self.conv5_1(x)
        return x
    
class VGG19_model():
  def __init__(self):
    selected_layers = ["block3_conv4", "block4_conv4", "block5_conv4"]
    self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    self.vgg.trainable = False
    self.outputs = [self.vgg.get_layer(l).output for l in selected_layers]
    
  def get_vgg19(self):
    vgg_model = tf.keras.Model(self.vgg.input, self.outputs)
    return vgg_model
