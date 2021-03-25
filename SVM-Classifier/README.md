# Requirement
 - python version = 3.7.10
```bash
$ pip3 install -r requirment.txt
```

# SVM-CLASSIFIER
SIFT 특징추출, BOW(Bag Of Word), 모델: SVM 사용
- 입력 이미지로부터 SIFT 특징을 추출한 후 KMeans 모델을 사용하여 BOW를 세운다. 각각의 이미지는 100차원의 벡터로 벡터화된다.
- train 시 SVM, KNN 모델을 사용하고 정확도를 비교한다.
- 강아지 비문을 분류하기위한 모델이다.

# 사용법
 - 테스트할 데이터를 디렉토리에 넣고 테스트파일과 디렉토리를 매개변수로 넣어서 실행
```bash
$ python Classifier.py --test test_0.jpg --dir Dog-Data
```

# Data Preprocess and augmentation
 - 데이터 이미지를 정규화한다. (preprocessing)
 - 정규화한 데이터 사이즈를 1/2, 1/3, 1/4 로 줄이고 저장한다. (data augmentation)

# 사용법
- 테스트할 데이터를 /image 디렉토리에 넣고 디렉토리이름을 매개변수로 넣어서 실행
```bash
$ python preprocess.py --dir 0
```
- 디렉토리에 있는 이미지를 전처리, 데이터 증강 후 /Dog-data/train 에 저장한다.