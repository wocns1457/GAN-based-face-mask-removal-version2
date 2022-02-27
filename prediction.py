# import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import noise_processing
from datasets import Dataset

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
    
    # local에서 1개 이미지 시각화하여 예측
    def one_predict(self, img_dir):        
        img = plt.imread(img_dir)
        if img_dir.endswith('.png'):
            img = img * 255.0
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img, [128, 128],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.reshape(img, [1, 128, 128, 3])
        img = tf.cast(img, tf.float32)
        img =  (img / 127.5) - 1
        
        mask = self.mask_model(img, training=False)
        mask = noise_processing(mask)
        face = self.face_model([img, mask], training=False)

        plt.figure(figsize=(7,7))
        plt.subplot(1, 2, 1)
        plt.title('Image with mask')
        plt.imshow(img[0] * 0.5 + 0.5)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title('Prediction Image')
        plt.imshow(face[0] * 0.5 + 0.5)
        plt.axis('off')
        plt.show()
        
    # local에서 여러개 이미지 시각화하여 예측
    def multiple_predict(self, img_dir):     
        testset = Dataset(file_path=img_dir, batch_size=1)
        testset = testset.make_test()
        img_num = len(testset)
        
        plt.figure(figsize=(10, 10))
        plt.suptitle('Prediction Image', fontsize=20, y=0.7)
        
        for i, img in enumerate(testset):
            mask = self.mask_model(img, training=False)
            process_img = noise_processing(img, mask)
            pred = self.face_model(process_img, training=False)
            pred = tf.concat([img, pred], axis=1)

            plt.subplot(1, img_num, i+1)
            plt.imshow(pred[0] * 0.5 + 0.5)
            plt.axis('off') 
        plt.show()
        
        
# Web app을 통한 예측
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
