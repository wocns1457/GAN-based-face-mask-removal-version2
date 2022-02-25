import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2

from train import *
from models import *
import numpy as np


plt.close()
plt.ioff()

def noise_processing(generate_image):
    generate_image = generate_image.numpy()
    batch, height, width  = generate_image.shape[0], generate_image.shape[1], generate_image.shape[2]
    generate_image = generate_image[:, :, :, 0]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    for i in range(batch):
        
        # generate_image = generate_image[i, :, :, :]
        generate_image[i]= cv2.erode(generate_image[i], k)             #  mask  Morphology 연산 전처리
        generate_image[i] = cv2.dilate(generate_image[i], k)

    generate_image = np.where(generate_image >= -0.9, 1, -1)
    generate_image = tf.convert_to_tensor(generate_image, dtype=tf.float32)
    generate_image = tf.reshape(generate_image, [batch, height, width , 1])
    return generate_image


def one_predict(mask_model, face_model, img_dir):        
    img = plt.imread(img_dir)
    if img_dir.endswith('.png'):
        img = img * 255.0
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, [128, 128],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.reshape(img, [1, 128, 128, 3])
    img = tf.cast(img, tf.float32)
    img =  (img / 127.5) - 1
    
    mask = mask_model(img, training=False)
    mask = noise_processing(mask)
    face = face_model([img, mask], training=False)

    plt.figure(figsize=(7,7))
    plt.subplot(1, 2, 1)
    plt.title('Image with mask')
    # plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    plt.imshow(img[0] * 0.5 + 0.5)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Prediction Image')
    # plt.imshow(tf.keras.preprocessing.image.array_to_img(face[0]))
    plt.imshow(face[0] * 0.5 + 0.5)
    plt.axis('off')
    plt.show()

PATH = os.getcwd()              
mask_checkpoint_dir = PATH+'/mask32_checkpoints'
face_checkpoint_dir = PATH+'/face_checkpoints'
dis_checkpoint_dir = PATH+'/dis_checkpoints'
test_dir = PATH + './testset/img5.jpeg'

with tf.device('/cpu:0'):
    m = Mask_G(filters=32)
    f = Face_G()
    w = Face_D_whole()
    r = Face_D_region()
    v = VGG19_model()

    train = Train_Face(m, f, w, r, v, mask_checkpoint_dir='./mask32_checkpoints', 
                   face_checkpoint_dir='./face_checkpoints', dis_checkpoint_dir='./dis_checkpoints')
    one_predict(train.mask_model, train.face_model, test_dir)
    
