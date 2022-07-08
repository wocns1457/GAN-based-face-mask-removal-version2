import argparse
import os
import tensorflow as tf
from datasets import Dataset
from models import Mask_G, Face_G, Face_D_whole, Face_D_region, VGG19_model
from train import Train_Mask, Train_Face
from prediction import Load_model
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Train the Mask removal network',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', default='multi_test', choices=['mask-train', 'face-train', 'single-test', 'multi-test'], dest='mode')
    parser.add_argument('--dir_test', default='./test', dest='dir_test')
    parser.add_argument('--m_ckpt_num', default=None, dest='m_ckpt_num')
    parser.add_argument('--f_ckpt_num', default=None, dest='f_ckpt_num')
    parser.add_argument('--d_ckpt_num', default=None, dest='d_ckpt_num')
    parser.add_argument('--choice_ckpt', dest='choice_ckpt',  action='store_true')
    parser.add_argument('--no-choice_ckpt', dest='choice_ckpt', action='store_false')
    parser.set_defaults(choice_ckpt=False)
    
    PATH = os.getcwd()
    train_path = PATH+'/train'                
    test_path = parser.parse_args().dir_test
    mask_checkpoint_dir = PATH+'/mask32_checkpoints'
    face_checkpoint_dir = PATH+'/face_checkpoints'
    dis_checkpoint_dir= PATH+'/dis_checkpoints'
    
    if parser.parse_args().mode == 'single-test':     # Visualize one image in a folder
        mask_G, face_G = Mask_G(filters=32), Face_G(filters=64)
        test = Load_model(mask_G, face_G, mask_checkpoint_dir=mask_checkpoint_dir, face_checkpoint_dir=face_checkpoint_dir)
        test.load()
        test.one_predict(test_path)
    elif parser.parse_args().mode == 'multi-test':    # Visualize up to 4 images in a folder
        mask_G, face_G = Mask_G(filters=32), Face_G(filters=64)
        test = Load_model(mask_G, face_G, mask_checkpoint_dir=mask_checkpoint_dir, face_checkpoint_dir=face_checkpoint_dir)
        test.load()
        test.multiple_predict(test_path)
    else:
        BATCH_SIZE = 8
        trainset = Dataset(file_path=train_path, batch_size=BATCH_SIZE)
        trainset = trainset.make_train()
        
        with tf.device('/gpu:0'):
            if parser.parse_args().mode == 'mask-train':
                mask_G= Mask_G(filters=32)
                mask_train = Train_Mask(mask_G, checkpoint_dir=mask_checkpoint_dir)
                if parser.parse_args().choice_ckpt :
                    mask_train.load(checkpoint_dir=mask_checkpoint_dir, ckpt_num=parser.parse_args().m_ckpt_num)
                mask_train.fit(trainset, epochs=5)
                
            elif parser.parse_args().mode == 'face-train':       
                mask_G, face_G, face_D_whole, face_D_region, vgg19 = Mask_G(filters=32), Face_G(filters=64), Face_D_whole(filters=64), Face_D_region(filters=64), VGG19_model()
                face_train = Train_Face(mask_G, face_G, face_D_whole, face_D_region, vgg19,
                                mask_checkpoint_dir=mask_checkpoint_dir, face_checkpoint_dir=face_checkpoint_dir, dis_checkpoint_dir=dis_checkpoint_dir)
                if parser.parse_args().choice_ckpt :
                    face_train.load(face_checkpoint_dir=face_checkpoint_dir, dis_checkpoint_dir=dis_checkpoint_dir, f_ckpt_num=parser.parse_args().f_ckpt_num, d_ckpt_num=parser.parse_args().d_ckpt_num)
                face_train.fit(trainset, epochs=5)

if __name__ == '__main__':
    main()
    
