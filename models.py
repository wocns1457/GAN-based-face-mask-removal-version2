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

    self.conv1_1 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.conv1_2 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn1_2 = BatchInstanceNormalization()
    self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
    
    self.conv2_1 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn2_1 = BatchInstanceNormalization()
    self.conv2_2 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn2_2 = BatchInstanceNormalization()
    self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
    
    self.conv3_1 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn3_1 = BatchInstanceNormalization()
    self.conv3_2 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn3_2 = BatchInstanceNormalization()
    self.pool3 = layers.MaxPooling2D(pool_size=(2, 2))
    
    self.conv4_1 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn4_1 = BatchInstanceNormalization()
    self.conv4_2 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn4_2 = BatchInstanceNormalization()
    self.pool4 = layers.MaxPooling2D(pool_size=(2, 2))
    
    self.conv5_1 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn5_1 = BatchInstanceNormalization()
    self.conv5_2 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn5_2 = BatchInstanceNormalization()
        
    self.convt1 = layers.Conv2DTranspose(filters*8, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.conv6_1 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn6_1 = BatchInstanceNormalization()
    self.conv6_2 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn6_2 = BatchInstanceNormalization()
    
    self.convt2 = layers.Conv2DTranspose(filters*4, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.conv7_1 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn7_1 = BatchInstanceNormalization()
    self.conv7_2 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn7_2 = BatchInstanceNormalization()
    
    self.convt3 = layers.Conv2DTranspose(filters*2, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.conv8_1 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn8_1 = BatchInstanceNormalization()
    self.conv8_2 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn8_2 = BatchInstanceNormalization()
    
    self.convt4 = layers.Conv2DTranspose(filters*1, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.conv9_1 = layers.Conv2D(filters*1, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn9_1 = BatchInstanceNormalization()
    self.conv9_2 = layers.Conv2D(filters*1, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn9_2 = BatchInstanceNormalization()
    
    self.out = layers.Conv2D(1, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)

  def call(self, inputs, training=False):
    # encoder
    enc_1 = self.conv1_1(inputs)
    enc_1 = self.conv1_2(enc_1)
    enc_1 = self.bn1_2(enc_1, training=training)
    enc_1 = tf.nn.leaky_relu(enc_1)
    pool1 = self.pool1(enc_1)

    enc_2 = self.conv2_1(pool1)
    enc_2 = self.bn2_1(enc_2, training=training)
    enc_2 = tf.nn.leaky_relu(enc_2)
    enc_2 = self.conv2_2(enc_2)
    enc_2 = self.bn2_2(enc_2, training=training)
    enc_2 = tf.nn.leaky_relu(enc_2)
    pool2 = self.pool2(enc_2)

    enc_3 = self.conv3_1(pool2)
    enc_3 = self.bn3_1(enc_3, training=training)
    enc_3 = tf.nn.leaky_relu(enc_3)
    enc_3 = self.conv3_2(enc_3)
    enc_3 = self.bn3_2(enc_3, training=training)
    enc_3 = tf.nn.leaky_relu(enc_3)
    pool3 = self.pool3(enc_3)

    enc_4 = self.conv4_1(pool3)
    enc_4 = self.bn4_1(enc_4, training=training)
    enc_4 = tf.nn.leaky_relu(enc_4)
    enc_4 = self.conv4_2(enc_4)
    enc_4 = self.bn4_2(enc_4, training=training)
    enc_4 = tf.nn.leaky_relu(enc_4)
    pool4 = self.pool4(enc_4)

    brige = self.conv5_1(pool4)
    brige = self.bn5_1(brige, training=training)
    brige = tf.nn.leaky_relu(brige)
    brige = self.conv5_2(brige)
    brige = self.bn5_2(brige, training=training)
    brige = tf.nn.leaky_relu(brige)

    # decoder
    dec_4 = self.convt1(brige)
    cat4 = layers.Concatenate()([enc_4, dec_4])
    dec_4 = self.conv6_1(cat4)
    dec_4 = self.bn6_1(dec_4)
    dec_4 = tf.nn.leaky_relu(dec_4)
    dec_4 = self.conv6_2(dec_4)
    dec_4 = self.bn6_2(dec_4)
    dec_4 = tf.nn.leaky_relu(dec_4)
    
    
    dec_3 = self.convt2(dec_4)
    cat3 = layers.Concatenate()([enc_3, dec_3])
    dec_3 = self.conv7_1(cat3)
    dec_3 = self.bn7_1(dec_3)
    dec_3 = tf.nn.leaky_relu(dec_3)
    dec_3 = self.conv7_2(dec_3)
    dec_3 = self.bn7_2(dec_3)
    dec_3 = tf.nn.leaky_relu(dec_3)
    
    dec_2 = self.convt3(dec_3)
    cat2 = layers.Concatenate()([enc_2, dec_2])
    dec_2 = self.conv8_1(cat2)
    dec_2 = self.bn8_1(dec_2)
    dec_2 = tf.nn.leaky_relu(dec_2)
    dec_2 = self.conv8_2(dec_2)
    dec_2 = self.bn8_2(dec_2)
    dec_2 = tf.nn.leaky_relu(dec_2)
    
    dec_1 = self.convt4(dec_2)
    cat1 = layers.Concatenate()([enc_1, dec_1])
    dec_1 = self.conv9_1(cat1)
    dec_1 = self.bn9_1(dec_1)
    dec_1 = tf.nn.leaky_relu(dec_1)
    dec_1 = self.conv9_2(dec_1)
    dec_1 = self.bn9_2(dec_1)
    dec_1 = tf.nn.leaky_relu(dec_1)
    
    out = self.out(dec_1)

    return tf.keras.activations.tanh(out)

class Face_G(tf.keras.Model):
  def __init__(self, filters=64, kernel_size=3, strides=1, padding='same'):
    super(Face_G, self).__init__()

    initializer = tf.random_normal_initializer(0., 0.02)
    self.strides = strides
    self.kernel_size = kernel_size
    self.padding = padding

    self.conv1_1 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.conv1_2 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
    
    self.conv2_1 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn2_1 = BatchInstanceNormalization()
    self.conv2_2 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn2_2 = BatchInstanceNormalization()
    self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))

    self.conv3_1 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, dilation_rate=(2, 2), use_bias=False)
    self.bn3_1 = BatchInstanceNormalization()
    self.conv3_2 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, dilation_rate=(4, 4), use_bias=False)
    self.bn3_2 = BatchInstanceNormalization()
    self.conv3_3 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, dilation_rate=(8, 8), use_bias=False)
    self.bn3_3 = BatchInstanceNormalization()
    self.conv3_4 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, dilation_rate=(16, 16), use_bias=False)
    self.bn3_4 = BatchInstanceNormalization()
        
    self.convt1 = layers.Conv2DTranspose(filters*2, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.conv6_1 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn6_1 = BatchInstanceNormalization()
    self.conv6_2 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn6_2 = BatchInstanceNormalization()
    
    self.convt2 = layers.Conv2DTranspose(filters*1, kernel_size=self.kernel_size, strides=self.strides*2, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.conv7_1 = layers.Conv2D(filters*1, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn7_1 = BatchInstanceNormalization()
    self.conv7_2 = layers.Conv2D(filters*1, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn7_2 = BatchInstanceNormalization()
    self.conv7_3 = layers.Conv2D(3, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn7_3 = BatchInstanceNormalization()
    
    self.out = layers.Conv2D(3, kernel_size=1, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    
  def call(self, inputs, training=False):
    # encoder
    input_mask = inputs[0]
    input_map = inputs[1]
    input = layers.concatenate([input_mask, input_map])
    
    enc_1 = self.conv1_1(input)
    enc_1 = self.conv1_2(enc_1)
    enc_1 = tf.nn.leaky_relu(enc_1)
    pool1 = self.pool1(enc_1)

    enc_2 = self.conv2_1(pool1)
    enc_2 = self.bn2_1(enc_2, training=training)
    enc_2 = tf.nn.leaky_relu(enc_2)
    enc_2 = self.conv2_2(enc_2)
    enc_2 = self.bn2_2(enc_2, training=training)
    enc_2 = tf.nn.leaky_relu(enc_2)
    pool2 = self.pool2(enc_2)

    enc_3 = self.conv3_1(pool2)
    enc_3 = self.bn3_1(enc_3, training=training)
    enc_3 = tf.nn.leaky_relu(enc_3)
    enc_3 = self.conv3_2(enc_3)
    enc_3 = self.bn3_2(enc_3, training=training)
    enc_3 = tf.nn.leaky_relu(enc_3)
    enc_3 = self.conv3_3(enc_3)
    enc_3 = self.bn3_3(enc_3, training=training)
    enc_3 = tf.nn.leaky_relu(enc_3)
    enc_3 = self.conv3_4(enc_3)
    enc_3 = self.bn3_4(enc_3, training=training)
    enc_3 = tf.nn.leaky_relu(enc_3)

    # decoder
    dec_4 = self.convt1(enc_3)
    cat4 = layers.Concatenate()([enc_2, dec_4])
    dec_4 = self.conv6_1(cat4)
    dec_4 = self.bn6_1(dec_4)
    dec_4 = tf.nn.leaky_relu(dec_4)
    dec_4 = self.conv6_2(dec_4)
    dec_4 = self.bn6_2(dec_4)
    dec_4 = tf.nn.leaky_relu(dec_4)
    
    dec_3 = self.convt2(dec_4)
    cat3 = layers.Concatenate()([enc_1, dec_3])
    dec_3 = self.conv7_1(cat3)
    dec_3 = self.bn7_1(dec_3)
    dec_3 = tf.nn.leaky_relu(dec_3)
    dec_3 = self.conv7_2(dec_3)
    dec_3 = self.bn7_2(dec_3)
    dec_3 = tf.nn.leaky_relu(dec_3)
    dec_3 = self.conv7_3(dec_3)
    dec_3 = self.bn7_3(dec_3)
    dec_3 = tf.nn.leaky_relu(dec_3)

    out = self.out(dec_3)

    return tf.keras.activations.tanh(out)

class Face_D_whole(tf.keras.Model):
    def __init__(self, filters=64, kernel_size=4, strides=1, padding='same'):
        super(Face_D_whole, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.filters = filters
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


class Face_D_whole(tf.keras.Model):
    def __init__(self, filters=64, kernel_size=4, strides=1, padding='same'):
        super(Face_D_whole, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.filters = filters
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

        self.filters = filters
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