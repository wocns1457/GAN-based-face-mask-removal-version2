import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

plt.ion()
figure, ax = plt.subplots(figsize=(8,8))

def noise_processing(generate_image):
    """
    Mask Generator를 통해 나온 Binary 이미지에 Noise 제거
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
        generate_image[i]= cv2.erode(generate_image[i], k)             #  mask Morphology 연산 전처리
        generate_image[i] = cv2.dilate(generate_image[i], k)
    generate_image = np.where(generate_image >= -0.9, 1, -1)
    generate_image = tf.convert_to_tensor(generate_image, dtype=tf.float32)
    generate_image = tf.reshape(generate_image, [batch, height, width , 1])
    return generate_image

def face_training_visualization(model, test_input, mask_map, tar, epoch, step, save_dir='./face32_training'):
    """
    fece model training visualization
    Args:
        model : Mask Generate model
        test_input : input image
        mask_map : Mask Generate model out
        tar : ground truth image
        epoch, step : training epoch, step
    """
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)

    prediction = model([test_input, mask_map], training=False)
    display_list = [test_input[0], tar[0], prediction[0], mask_map[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image', 'Predicted mask']

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(save_dir+"/epoch_{epoch}_step_{step}.png".format(epoch=epoch, step=step))
    plt.show()

    figure.canvas.draw()
    figure.canvas.flush_events()
