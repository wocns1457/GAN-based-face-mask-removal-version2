# import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model import *

import warnings
warnings.filterwarnings('ignore')

class Load_model():
    def __init__(self, mask_model, face_model, mask_checkpoint_dir, face_checkpoint_dir):
        self.mask_model = mask_model
        self.face_model = face_model
        self.mask_model.build(input_shape=(None, 128, 128, 3))
        self.face_model.build(input_shape=[(None, 128, 128, 3), (None, 128, 128, 1)])  
        
        self.mask_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.face_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        self.mask_checkpoint_dir = mask_checkpoint_dir
        self.face_checkpoint_dir = face_checkpoint_dir
        

        self.mask_checkpoint = tf.train.Checkpoint(generator_optimizer=self.mask_optimizer,
                                            generator=self.mask_model)        
        self.face_checkpoint = tf.train.Checkpoint(generator_optimizer=self.face_optimizer,
                                            generator=self.face_model)
        
    def load(self):
        self.mask_checkpoint.restore(tf.train.latest_checkpoint(self.mask_checkpoint_dir))
        self.face_checkpoint.restore(tf.train.latest_checkpoint(self.face_checkpoint_dir))
    
# def noise_processing(generate_image):
#     generate_image = generate_image.numpy()
#     batch, height, width  = generate_image.shape[0], generate_image.shape[1], generate_image.shape[2]
#     generate_image = generate_image[:, :, :, 0]
#     k = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
#     for i in range(batch):
#         # generate_image = generate_image[i, :, :, :]
#         generate_image[i]= cv2.erode(generate_image[i], k)             #  mask  Morphology 연산 전처리
#         generate_image[i] = cv2.dilate(generate_image[i], k)

#         generate_image = np.where(generate_image >= -0.9, 1, -1)
#         generate_image = tf.convert_to_tensor(generate_image, dtype=tf.float32)
#         generate_image = tf.reshape(generate_image, [batch, height, width , 1])
#         return generate_image

def pred(img_dir, mask_model, face_model):
    img = plt.imread(img_dir)
    img_name = img_dir.split('/')[-1]

    height, width = img.shape[0], img.shape[1]
    if img_dir.endswith('.png') or img_dir.endswith('.PNG'):
        img = img * 255.0
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, [128, 128],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.reshape(img, [1, 128, 128, 3])
    img = tf.cast(img, tf.float32)
    img =  (img / 127.5) - 1
    
    mask = mask_model(img, training=False)
    # mask = noise_processing(mask)
    face = face_model([img, mask], training=False)
    face = face[0]
    face = tf.image.resize(face, [height, width],
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    tf.keras.preprocessing.image.save_img("flask_app/static/prediction/"+img_name, face, scale=True)
        

