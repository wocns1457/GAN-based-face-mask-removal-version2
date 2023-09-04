import matplotlib.pyplot as plt
import tensorflow as tf
from utils import noise_processing
from datasets import Dataset

class Load_Model():
    def __init__(self, mask_model, face_model, mask_checkpoint_dir, face_checkpoint_dir):
        self.mask_model = mask_model
        self.face_model = face_model
        
        self.mask_checkpoint_dir = mask_checkpoint_dir
        self.face_checkpoint_dir = face_checkpoint_dir

        self.mask_checkpoint = tf.train.Checkpoint(generator=self.mask_model)        
        self.face_checkpoint = tf.train.Checkpoint(generator=self.face_model)
        
        self.mask_checkpoint.restore(tf.train.latest_checkpoint(self.mask_checkpoint_dir))
        self.face_checkpoint.restore(tf.train.latest_checkpoint(self.face_checkpoint_dir))
    
    def predict(self, img_path, img_size=256):        
        img = plt.imread(img_path)
        if img_path.endswith('.png') or img_path.endswith('.PNG'):
            img = img * 255.0
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img, [img_size, img_size],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.reshape(img, [1, img_size, img_size, 3])
        img = tf.cast(img, tf.float32)
        img =  (img / 127.5) - 1
        
        mask = self.mask_model(img, training=False)
        mask = noise_processing(mask)
        pred = self.face_model([img, mask], training=False)

        plt.figure(figsize=(8,8))
        plt.title('Prediction Image')
        plt.imshow(pred[0] * 0.5 + 0.5)
        plt.axis('off')
        plt.show()
