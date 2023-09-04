import os
import argparse
import tensorflow as tf
from datasets import Dataset
from models import Mask_G, Face_G, Face_D_whole, Face_D_region, VGG19_model
from train import Train_Model
from prediction import Load_Model
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Train the Mask removal network',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', default='test', choices=['train', 'test'], dest='mode')
    parser.add_argument('--epochs', default=50, dest='epochs', type=int)
    parser.add_argument('--batch', default=8, dest='batch', type=int)
    parser.add_argument('--test_path', default='./test', dest='test_path')
    parser.add_argument('--m_ckpt_num', default=None, dest='m_ckpt_num')
    parser.add_argument('--f_ckpt_num', default=None, dest='f_ckpt_num')
    parser.add_argument('--d_ckpt_num', default=None, dest='d_ckpt_num')
    parser.add_argument('--choice_ckpt', dest='choice_ckpt',  action='store_true')
    parser.add_argument('--no-choice_ckpt', dest='choice_ckpt', action='store_false')
    parser.set_defaults(choice_ckpt=False)
    args = parser.parse_args()

    PATH = os.getcwd()
    train_path = PATH+'/train'                
    test_path = args.test_path
    mask_checkpoint_dir = PATH+'/mask_checkpoints'
    face_checkpoint_dir = PATH+'/face_checkpoints'
    dis_checkpoint_dir= PATH+'/dis_checkpoints'
    
    if args.mode == 'train':
        trainset = Dataset(file_path=train_path, batch_size=args.batch)
        trainset = trainset.make_train()
        
        with tf.device('/gpu:0'):     
            mask_G, face_G, face_D_whole, face_D_region, vgg19 = Mask_G(filters=32), Face_G(filters=32), Face_D_whole(filters=64), Face_D_region(filters=64), VGG19_model()
            train = Train_Model(mask_G, face_G, face_D_whole, face_D_region, vgg19, 
                                mask_checkpoint_dir=mask_checkpoint_dir, 
                                face_checkpoint_dir=face_checkpoint_dir, 
                                dis_checkpoint_dir=dis_checkpoint_dir)
            
            if args.choice_ckpt:
                train.load(m_ckpt_num=args.m_ckpt_num,
                           f_ckpt_num=args.f_ckpt_num, 
                           d_ckpt_num=args.d_ckpt_num)
                
            train.fit(trainset, epochs=args.epochs)

    if args.mode == 'test': 
        mask_G, face_G = Mask_G(filters=32), Face_G(filters=32)
        test = Load_Model(mask_G, face_G, 
                          mask_checkpoint_dir=mask_checkpoint_dir, 
                          face_checkpoint_dir=face_checkpoint_dir)
        test.predict(test_path)

if __name__ == '__main__':
    main()
