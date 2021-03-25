import os
import argparse
import cv2
import numpy as np

from PIL import Image
from histo_clahe import histo_clahe

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--dir', default='0',help='dataset directory')
parser.add_argument('--savedir', default='./Dog-Data/train',help='save directory')
opt = parser.parse_args()

path = './image/' + opt.dir
file_list = os.listdir(path)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


for file in file_list:
    s = os.path.splitext(file)
    savedir = ['0']
    createFolder(opt.savedir + '/' + opt.dir)
    for i in range(1,5):
        savedir.append(opt.savedir + '/' + opt.dir + '/' + s[0] + '-' + str(i) + s[1])

    img = histo_clahe(path + '/' + file)
    cv2.imwrite(savedir[1],img) 

    width = img_clahe_save.shape[1]
    height = img_clahe_save.shape[0]

    img_resize = cv2.resize(img,(int(width / 2), int(height / 2))) 
    cv2.imwrite(savedir[2],img_resize)
    img_resize = cv2.resize(img,(int(width / 3), int(height / 3))) 
    cv2.imwrite(savedir[3],img_resize)
    img_resize = cv2.resize(img,(int(width / 4), int(height / 4))) 
    cv2.imwrite(savedir[4],img_resize)
