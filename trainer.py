import os
import tensorflow as tf

from datasets import Dataset
from models import Mask_G, Face_G, Face_D_whole, Face_D_region, VGG19_model
from train import Train_Face


with tf.device('/cpu:0'):     
    m = Mask_G(filters=32)
    f = Face_G()
    w = Face_D_whole()
    r = Face_D_region()
    v = VGG19_model()
    face_checkpoint_dir='./face_checkpoints'
    dis_checkpoint_dir='./dis_checkpoints'
    train = Train_Face(m, f, w, r, v, mask_checkpoint_dir='./mask32_checkpoints', face_checkpoint_dir=face_checkpoint_dir, dis_checkpoint_dir=dis_checkpoint_dir)

    BATCH_SIZE = 4
    train_path = './train'
    trainset = Dataset(file_path=train_path, batch_size=BATCH_SIZE, task='face')
    trainset = trainset.make_train()

    train.fit(trainset, 10)
