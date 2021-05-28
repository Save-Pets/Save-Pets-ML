import os
import numpy as np
import cv2

import pickle
import sklearn
import argparse
import time

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from histo_clahe import histo_clahe

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--dir', default='Dog-Data',help='dataset directory')
parser.add_argument('--test', default='test_1.jpg',help='test image data')
opt = parser.parse_args()

#read data
def read_data(label2id):
    X = []
    Y = []
    for label in os.listdir('Dog-Data/train'):
        if os.path.isdir(os.path.join('Dog-Data/train', label)):
            for img_file in os.listdir(os.path.join('Dog-Data/train', label)):
                img = cv2.imread(os.path.join('Dog-Data/train', label, img_file))
                X.append(img)
                Y.append(label2id[label])
    return X, Y

# feature exraction with SIFT
def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.xfeatures2d.SIFT_create(nfeatures =200, nOctaveLayers =3, contrastThreshold=0.0005)

    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)

    return image_descriptors

#Kmeans bow
def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_
    return bow_dict

#image to vector.
def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []
    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance, axis=1)
            for j in argmin:
                features[j] += 1
        X_features.append(features)
    return X_features

def main():
    start = time.time()
    # Label to id
    path = "Dog-Data/train"
    file_list = os.listdir(path)
    label2id = {}

    idx = 0
    for i, label in enumerate(file_list):
        if label == ".DS_Store": 
            continue;
        label2id[label] = idx
        idx += 1
        
    X, Y = read_data(label2id)

    image_descriptors = extract_sift_features(X)

    # all descriptors
    all_descriptors = []
    for descriptors in image_descriptors:
        if descriptors is not None:
            for des in descriptors:
                all_descriptors.append(des)


    num_clusters = 100

    if not os.path.isfile('Dog-Data/bow.pkl'):
        BoW = kmeans_bow(all_descriptors, num_clusters)
        pickle.dump(BoW, open('Dog-Data/bow.pkl', 'wb'))
    else:
        BoW = pickle.load(open('Dog-Data/bow.pkl', 'rb'))

    X_features = create_features_bow(image_descriptors, BoW, num_clusters)

    #Set model
    X_train = [] 
    X_test = []
    Y_train = []
    Y_test = []
    X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.2, random_state=42, shuffle=True)

    #Train SVM
    svm = sklearn.svm.SVC(C = 100, probability=True)
    svm.fit(X_train, Y_train)

    #Train KNN
    knn = KNeighborsClassifier(n_neighbors=10,  weights='distance', leaf_size=200, p=2)
    knn.fit(X_train, Y_train)

    #predict
    #img_test = cv2.imread(opt.dir + '/test/' + opt.test)
    img_test = histo_clahe('Dog-Data/test/' + opt.test)

    img = [img_test]
    img_sift_feature = extract_sift_features(img)
    img_bow_feature = create_features_bow(img_sift_feature, BoW, num_clusters)

    #---------------------------------
    #-----------evaluation------------
    #---------------------------------

    #predict SVM
    img_predict = svm.predict(img_bow_feature)
    #predict KNN
    img_predict2 = knn.predict(img_bow_feature)

    #prediction probability
    svm_prob = svm.predict_proba(img_bow_feature)[0][img_predict[0]]
    knn_prob = knn.predict_proba(img_bow_feature)[0][img_predict2[0]]

    # print("SVM prob: ", svm.predict_proba(img_bow_feature))
    # print("KNN prob: ", knn.predict_proba(img_bow_feature))
    
    for key, value in label2id.items():
        if value == img_predict[0]:
            svm_k = key
        if value == img_predict2[0]:
            knn_k = key

    result =""
    if svm_k==knn_k:
        result = result+svm_k+","
    else:
        result = result+"a,"
    # result = result+"202151796꿍1234"+","
    # result =result+"등록된강아지"+","

    if (svm_prob < 0.65 and knn_prob < 0.55) or svm_k != knn_k:
        result = result+"미등록강아지"+","
    else:
        result = result+"등록된강아지"+","
    #Accuracy
    if svm.score(X_test,Y_test) > knn.score(X_test,Y_test):
        result = result+str(svm.score(X_test,Y_test))
    else :
        result = result+str(knn.score(X_test,Y_test))
        
    # print("SVM Score: ", svm.score(X_test, Y_test))
    # print("KNN Score: ", knn.score(X_test, Y_test))
    # print("running time: ", round(time.time() - start, 2))

    print(result)

if __name__ == "__main__":
    main()