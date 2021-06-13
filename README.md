# 구해줘 펫즈, Save Pets

</br>

### 1. 데모 영상

[데모 영상 보러가기](https://user-images.githubusercontent.com/20268101/120226192-b0571980-c281-11eb-9c59-8288b7d655c1.mp4)

### 2. 서비스 소개

![savepets_main](https://user-images.githubusercontent.com/20268101/120218227-60258a80-c274-11eb-8f81-2abcfc561f43.png)

### 3. 주요 기술 및 설명

#### [Client]
</br>
* **iOS**
  * UIKit
  * GCD (DispatchSemaphore)
  * AVFoundation
  * Vision
  * CoreML
  </br>

#### [Back end]
* **Server**
  * Flask
  * MySQL
  * AWS
* **Machine Learning**
  * OpenCV
  * Scikit learn
  * Pytorch
  </br>




#### 1) 강아지 코를 찾기 위한 객체 탐지 모델 - YOLOv5

비문으로 강아지를 분류하려면 먼저 강아지 얼굴에서 강아지 코를 찾을 수 있어야합니다.  

물론 클라이언트로부터 강아지 코를 지정하여 입력받을 수도 있지만 사용자가 엉뚱한 사진으로 강아지를 등록하는 것을 방지하기 위해
검증된 강아지 코만 입력받아야 합니다.  

때문에 강아지 코를 찾기위한 객체 탐지(Object Detection) 모델이 필요합니다.

객체 탐지 분야에서 유명한 YOLO(You Only Look Once)중 [YOLOv5](https://github.com/ultralytics/yolov5) 모델을 사용하여 강아지 코 탐지 모델을 구현하였습니다.

[Stanford의 Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) 중 해상도가 높은 사진을 3060장 골라 직접 라벨링하여 모델을 학습하였습니다.

<img src="https://user-images.githubusercontent.com/39593640/121805255-6c9edf80-cc85-11eb-8aff-1dd8892deeee.PNG" alt="Original" style="zoom: 67%;" />

<img src="https://user-images.githubusercontent.com/39593640/121805263-70cafd00-cc85-11eb-9c76-a8cfd7e197dd.PNG" alt="YOLOv5" style="zoom: 67%;" />



</br>

#### 2) 이미지 전처리 - CLAHE

강아지 코를 촬영할 때 그늘지면 주름이 잘 보이지 않습니다.

주름 무늬가 잘 보이도록 어두운 부분을 밝게 펴주는 전처리가 필요합니다.  

CLAHE는 히스토그램 높이에 제한을 둬서 특정 높이 이상에 있는 pixel 값들을 재분배하는 이미지 평탄화 방식입니다.

<img src="https://user-images.githubusercontent.com/39593640/121805265-73c5ed80-cc85-11eb-9d7c-c668705fe643.png" alt="CLAHE" />

CLAHE를 적용 전 이미지

![original image](https://user-images.githubusercontent.com/39593640/121805267-74f71a80-cc85-11eb-9bf7-978885b411dd.PNG)

CLAHE를 적용 후 이미지

![CLAHE image](https://user-images.githubusercontent.com/39593640/121805269-76284780-cc85-11eb-8028-01ccebfd868b.PNG)

</br>

#### 3) 특징 추출 및 벡터화 - SIFT, K-Means

이미지로만 분류한다면 처리해야하는 텐서의 차원이 높아지고 높은 차원의 입력을 처리하기 위해서는 복잡한 모델이 요구됩니다.

예를들어 100x100x3 이미지라면 30000개의 특징들이 존재하게 됩니다.

이미지 분류에 있어서 이미지의 크기와 필요한 학습 데이터 수는 비례합니다. 

하지만 사용자에게 100장 이상의 강아지 사진을 요구할 수는 없습니다.

따라서 입력의 차원을 줄여주는 전처리 과정이 필요합니다.

먼저 평탄화한 이미지에서 특징을 추출합니다. 특징 추출 알고리즘은 SIFT를 사용합니다.

그 후 추출한 특징을 100차원 벡터로 변환하여 분류합니다. 벡터화 알고리즘은 K-means Bag of Words 기법으로 구현하였습니다.

![sift](https://user-images.githubusercontent.com/39593640/121805289-8cce9e80-cc85-11eb-98bd-ffd7c0f3fbb2.PNG)

</br>

#### 4) 이미지 분류를 위한 분류 모델 - SVM

100차원의 벡터로 변환된 이미지를 SVM 모델을 이용하여 분류합니다.

Scikit leran의 SVM을 이용하여 분류하였습니다. 






### 4. UI/UX

#### 1) 비문 등록하기 (카메라/앨범선택)

|                                                              |                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![120216339-dbd20800-c271-11eb-80ee-12759a44ca35](https://user-images.githubusercontent.com/20268101/120216339-dbd20800-c271-11eb-80ee-12759a44ca35.png) | ![120221033-aaa90600-c278-11eb-8faa-5c04c2d6423c](https://user-images.githubusercontent.com/20268101/120221033-aaa90600-c278-11eb-8faa-5c04c2d6423c.png) | ![120216337-db397180-c271-11eb-9bba-159e554b4b57](https://user-images.githubusercontent.com/20268101/120216337-db397180-c271-11eb-9bba-159e554b4b57.png) |
| ![120222169-7cc4c100-c27a-11eb-8829-cb0041a7523e](https://user-images.githubusercontent.com/20268101/120222169-7cc4c100-c27a-11eb-8829-cb0041a7523e.png) | ![120216333-daa0db00-c271-11eb-8731-f9aac8700b00](https://user-images.githubusercontent.com/20268101/120216333-daa0db00-c271-11eb-8731-f9aac8700b00.png) |                                                              |
| ![120221032-a977d900-c278-11eb-90d1-5e506d9698e2](https://user-images.githubusercontent.com/20268101/120221032-a977d900-c278-11eb-90d1-5e506d9698e2.png) | ![120221031-a8df4280-c278-11eb-9006-e3f1b7b130b9](https://user-images.githubusercontent.com/20268101/120221031-a8df4280-c278-11eb-9006-e3f1b7b130b9.png) | ![120221025-a7ae1580-c278-11eb-9304-5273ac9b0523](https://user-images.githubusercontent.com/20268101/120221025-a7ae1580-c278-11eb-9304-5273ac9b0523.png) |
| ![120221008-a2e96180-c278-11eb-8ee0-fdcbd3a0805c](https://user-images.githubusercontent.com/20268101/120221008-a2e96180-c278-11eb-8ee0-fdcbd3a0805c.png) | ![120216326-da084480-c271-11eb-86f6-ea12f123956b](https://user-images.githubusercontent.com/20268101/120216326-da084480-c271-11eb-86f6-ea12f123956b.png) | ![120216323-d83e8100-c271-11eb-91b8-5e93db02a112](https://user-images.githubusercontent.com/20268101/120216323-d83e8100-c271-11eb-91b8-5e93db02a112.png) |
| ![120216311-d379cd00-c271-11eb-8aad-ccb193881ea1](https://user-images.githubusercontent.com/20268101/120216311-d379cd00-c271-11eb-8aad-ccb193881ea1.png) | ![120216353-deccf880-c271-11eb-8170-25b6087b977d](https://user-images.githubusercontent.com/20268101/120216353-deccf880-c271-11eb-8170-25b6087b977d.png) | ![120221828-058f2d00-c27a-11eb-8db7-3517c4f582b6](https://user-images.githubusercontent.com/20268101/120221828-058f2d00-c27a-11eb-8db7-3517c4f582b6.png) |



#### 2) 비문 조회하기 (카메라/앨범선택)

|                                                              |                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![120216339-dbd20800-c271-11eb-80ee-12759a44ca35](https://user-images.githubusercontent.com/20268101/120216339-dbd20800-c271-11eb-80ee-12759a44ca35.png) | ![120221926-29eb0980-c27a-11eb-8ac1-adf1483d1f54](https://user-images.githubusercontent.com/20268101/120221926-29eb0980-c27a-11eb-8ac1-adf1483d1f54.png) | ![120222169-7cc4c100-c27a-11eb-8829-cb0041a7523e](https://user-images.githubusercontent.com/20268101/120222169-7cc4c100-c27a-11eb-8829-cb0041a7523e.png) |
| ![120216325-d96fae00-c271-11eb-987d-d46f2f4755d6](https://user-images.githubusercontent.com/20268101/120216325-d96fae00-c271-11eb-987d-d46f2f4755d6.png) | ![120221937-2eafbd80-c27a-11eb-83df-81017923a218](https://user-images.githubusercontent.com/20268101/120221937-2eafbd80-c27a-11eb-83df-81017923a218.png) |                                                              |
| ![120216334-db397180-c271-11eb-94e2-da8e3977df19](https://user-images.githubusercontent.com/20268101/120216334-db397180-c271-11eb-94e2-da8e3977df19.png) | ![120216330-da084480-c271-11eb-926e-e55d350b4a5e](https://user-images.githubusercontent.com/20268101/120216330-da084480-c271-11eb-926e-e55d350b4a5e.png) | ![120222091-60288900-c27a-11eb-9ea8-89e6f25641db](https://user-images.githubusercontent.com/20268101/120222091-60288900-c27a-11eb-9ea8-89e6f25641db.png) |



