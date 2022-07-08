# GAN-based-face-mask-removal-version2
이 프로젝트는 마스크를 쓴 인물의 이미지에서 마스크 뒤에 감춰진 얼굴을 복원하는 프로젝트로, 
GAN-based-face-mask-removal-version1에서의 문제점인 일반화 성능이 낮다는 문제를 해결하기 위한 프로젝트입니다.
U-Net구조를 기반으로 하여 마스크 객체의 값을 1로 나머지 부분을 0으로 [Segmentation](https://arxiv.org/abs/1505.04597)하여 Binary map을 도출하고,
마스크를 쓴 인물의 사진과, Binary map을 [GAN](https://arxiv.org/abs/1406.2661)구조를 사용해 마스크 영역을 제거하여 제거된 부분을 실제 얼굴로 복원합니다.
그리고 Heroku를 통해 모델을 배포하여 Web을 통해 모델을 테스트를 진행해 볼 수 있습니다.  

http://mask-removal.herokuapp.com/  
[발표 영상 링크](https://drive.google.com/file/d/1cW06QysHOYnOB2aCgz3t5ItglEWY8FOz/view?usp=sharing)

## Environment
- python==3.8
- tensorflow==2.5.0
- opencv-python==4.5.5.62
- dlib==19.18
- numpy==1.22.0
- matplotlib==3.5.1

## DataSet
학습에 사용된 이미지는 약 50,000장 정도 사용되었습니다.
- CelebA dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Data Preprocessing
학습에 적합한 데이터를 사용하기 위해서 CelebA원본 Data에 마스크 이미지를 합성해줍니다.
version1에서는 얼굴의 각도에 따른 마스크의 위치를 고려하지 못했지만, 
이번 프로젝트에서는 아래와 같은 방법으로 얼굴의 각도에 따른 마스크를 합성을 통해 더 자연스러운 데이터를 전처리 하였습니다.
<p align="center">
<img src = "https://github.com/wocns1457/GAN-based-face-mask-removal-version2/blob/main/sample_img/Preprocessing.JPG" width="75%" height="75%">
</p>

## Model Architecture
<p align="center">
<img src = "https://github.com/wocns1457/GAN-based-face-mask-removal-version2/blob/main/sample_img/model.JPG" width="70%" height="70%">
</p>

마스크를 쓴 인물의 이미지를 입력으로 받아 입력받은 이미지에서 마스크에 해당하는 영역을 Segmentation하기 위해 U-Net을 기반으로 한 Mask G모델을 통과하여 마스크 부분에 해당하는 이진 맵을 아웃풋으로 얻습니다.

그 후 원래의 인풋 이미지와 Mask G에서 나온 Binary map을 채널 방향으로 concat시켜준 후 Face G모델에 입력해 주게 됩니다.
Face D - whole 은 생성된 이미지 전체를 바라보면서 가짜 이미지인지 진짜 이미지 인지 판별하게 되고 Face D – region 은 생성된 이미지에서 마스크 부분에 해당하는 영역을 바라보면서 가짜인지 진짜인지 판별하는 모델입니다.

생성된 이미지와 정답 이미지를 pre-train된 VGG19 network를 사용하여 이 네트워크를 통과한 Feature map 에서 픽셀값이 아닌 둘 사이의 유사도를 계산하여 더 디테일하게 이미지를 복원합니다.

  - Batch size:8
  - Training epoch:10
  - Adam optimizer
  - Learning rate:0.0002
  - Leaky ReLU:0.2

## How to Run

- Mask G Training
 ```
 python main.py --mode mask-train  # If a checkpoint exists, training proceeds from the latest checkpoint.
 python main.py --mode mask-train --choice_ckpt --m_ckpt_num 'num'  # If you want to train at a specific checkpoint.
 ```
- Face G Training, Face D whole, region Training
 ```
 python main.py --mode face-train  # If a checkpoint exists, training proceeds from the latest checkpoint.
 python main.py --mode face-train --choice_ckpt --f_ckpt_num 'num' --m_ckpt_num 'num'  # If you want to train at a specific checkpoint.
 ```
- Test
 ```
 python main.py --mode multi-test --dir_test ./test  # Make predictions from the latest checkpoints.
 ```
 - Wep test
 ```
export FLASK_APP=flask_app
flask run
 ```
 
 ## Result
 - Performance evaluation  
 CelebA 데이터중 10,000장의 이미지를 Test data로 사용하였고, 실제 이미지와 생성된 이미지 간의 특징 거리 측정에 가장 널리 사용되는 평가 지표인 FID(Frechet Inception Distance)를 사용하였습니다.  
 
    >***Version1 FID score : 22.863***  
    >***Version2 FID score : 28.206***
    
<br>

- Version1, 2 결과 비교
<p align="center">
<img src = "https://github.com/wocns1457/GAN-based-face-mask-removal-version2/blob/main/sample_img/result2.JPG" width="45%" height="50%">
</p>  
version2의 score가 version1보다 높게 나왔지만, 이는 Test data의 한정으로 나온 것이기 때문에 version1의 모델이 version2의 모델보다 일반화 성능이 좋다고 판단할 수 없습니다.
하지만 이번 프로젝트에서는 새로운 데이터에 대해서 마스크가 비교적 잘 제거되고 해당 부분에 얼굴이 생성된 것을 볼 수 있습니다.

## Reference
- NIZAM UD DIN, KAMRAN JAVED , SEHO BAE, AND JUNEHO YI "A Novel GAN-Based Network for Unmasking of Masked Face"  
  (pull paper : https://ieeexplore.ieee.org/document/9019697)
 - Data Preprocessing : https://github.com/aqeelanwar/MaskTheFace
- Instance Normalization code : https://github.com/bigbreadguy/Batch_Instance_Normalization-Tensorflow2.0
- FID score for PyTorch : https://github.com/mseitzer/pytorch-fid
