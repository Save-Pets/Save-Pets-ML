import cv2
import numpy as np

def histo_clahe(img_dir):
    #--1이미지 읽어서 YUV 컬러스페이스로 변경
    img = cv2.imread(img_dir)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    #--2 밝기 채널에 대해서 이퀄라이즈 적용
    img_eq = img_yuv.copy()
    img_eq[:,:,0] = cv2.equalizeHist(img_eq[:,:,0])
    img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

    #--3 밝기 채널에 대해서 CLAHE 적용
    img_clahe = img_yuv.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
    img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])           #CLAHE 적용
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

    return img_clahe[:,:,:]