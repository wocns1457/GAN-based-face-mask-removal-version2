import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

plt.ion()
figure, ax = plt.subplots(figsize=(8,8))

np.random.seed(42)

def noise_processing(generate_image):
    """
    Mask Generator를 통해 나온 Binary 이미지에 원본 이미지를 합성하여 Noise 생성
    Args:
      generate_image : model를 통해 생성된 이미지
    Return:
      processe가 된 인물 이미지
    """
    generate_image = generate_image.numpy()
    batch, height, width  = generate_image.shape[0], generate_image.shape[1], generate_image.shape[2]
    generate_image = generate_image[:, :, :, 0]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    for i in range(batch):
        generate_image[i]= cv2.erode(generate_image[i], k)             #  mask  Morphology 연산 전처리
        generate_image[i] = cv2.dilate(generate_image[i], k)
    generate_image = np.where(generate_image >= -0.9, 1, -1)
    generate_image = tf.convert_to_tensor(generate_image, dtype=tf.float32)
    generate_image = tf.reshape(generate_image, [batch, height, width , 1])
    return generate_image

def training_visualization(model, test_input, tar, epoch, step):
    """
    training visualization
    Args:
        model : Generate model
    """
    
    save_dir = './mask32_training'
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)  

    prediction = model(test_input, training=False)
    display_list = [test_input[0], tar[0], prediction[0, :, :, 0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
        
    plt.savefig(save_dir+"/epoch_{epoch}_step_{step}.png".format(epoch=epoch+1, step=step+1))
    plt.show()
    
    figure.canvas.draw()
    figure.canvas.flush_events()


def face_training_visualization(model, test_input, binary_input, tar, epoch, step):
    """
    training visualization
    Args:
        model : Generate model
    """
    
    save_dir = './face32_training'
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)  
      
    prediction = model([test_input, binary_input], training=False)
    print(prediction.shape)
    display_list = [test_input[0], tar[0], prediction[0, :, :, :]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(save_dir+"/epoch_{epoch}_step_{step}.png".format(epoch=epoch+1, step=step+1))
    plt.show()
    
    figure.canvas.draw()
    figure.canvas.flush_events()
    
    
def generate_images(model, test_input, tar):
    """
    Generate된 이미지 시각화
    Args:
        model : Generate model
    """
    prediction = model(test_input, training=False)
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()