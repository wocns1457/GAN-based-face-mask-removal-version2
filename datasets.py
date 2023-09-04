import os
import tensorflow as tf

class Dataset():
    def __init__(self, file_path, batch_size, img_size=256):
        self.image_file = [file_path +'/'+filename for filename in os.listdir(file_path)]
        self.batch_size = batch_size
        self.img_size = img_size

    def train_image(self, image_file):
        """
        train image load, 224x672이미지 real_image, mask_image, binary_image 분할
        """
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        image = tf.io.decode_image(image, channels=3, expand_animations = False)

        w = tf.shape(image)[1]
        w = w // 3

        real_image = image[:, :w, :]
        mask_image = image[:, w:w+w, :]
        binary_image = image[:, w+w:, :]
        # binary_image = binary_image[:, :, 0]

        # Convert both images to float32 tensors
        real_image = tf.cast(real_image, tf.float32)
        mask_image = tf.cast(mask_image, tf.float32)
        binary_image = tf.cast(binary_image, tf.float32)
        return real_image, mask_image, binary_image

    def test_image(self, image_file):
        """
        test image load
        """
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        image = tf.io.decode_image(image, channels=3, expand_animations = False)

        input_image = image[:, :, :]

        # Convert both images to float32 tensors
        input_image = tf.cast(input_image, tf.float32)
        return input_image

    def train_resize_and_normalize(self, real_image, mask_image, binary_image, height, width):
        """
        image resize and [-1 ~ 1] normalize
        """
        real_image = tf.image.resize(real_image, [height, width],
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask_image = tf.image.resize(mask_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        binary_image = tf.image.resize(binary_image, [height, width],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        real_image = (real_image / 127.5) - 1
        mask_image = (mask_image / 127.5) - 1
        binary_image = (binary_image / 127.5) - 1
        binary_image = tf.where(binary_image > -0.9, 1.0, -1.0)
        return real_image, mask_image, binary_image

    def test_resize_and_normalize(self, mask_image, height, width):
        """
        image resize and [-1 ~ 1] normalize
        """
        input_image = tf.image.resize(mask_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        input_image = tf.cast(input_image, tf.float32)
        input_image = (input_image / 127.5) - 1
        return input_image

    def load_image_train(self, image_file):
        real_image, mask_image, binary_image = self.train_image(image_file)
        real_image, mask_image, binary_image = self.train_resize_and_normalize(real_image, mask_image, binary_image, self.img_size, self.img_size)
        return real_image, mask_image, binary_image

    def load_image_test(self, image_file):
        input_image = self.test_image(image_file)
        input_image = self.test_resize_and_normalize(input_image, self.img_size, self.img_size)
        return input_image

    def make_train(self):
        dataset = tf.data.Dataset.list_files(self.image_file, shuffle=True, seed=42)
        dataset = dataset.map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def make_test(self):
        dataset = tf.data.Dataset.list_files(self.image_file, shuffle=False)
        dataset = dataset.map(self.load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        return dataset
