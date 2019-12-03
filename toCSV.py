import cv2
import os
import dlib
from skimage import io
import csv
import numpy as np


#所有图片的路径
path_image_from_camera = "data/data_faces_from_camera"

#探测器？如何实现？
detector = dlib.get_frontal_face_detector()

#特征检测仪，data/dlib/shape_predictor_5_face_landmarks.dat为训练好的数据
predictor = dlib.shape_predictor("data/dlib/shape_predictor_5_face_landmarks.dat")
#？
face_rec = dlib.face_recognition_model_v1("data/dlib/dlib_face_recognition_resnet_model_v1.dat")

#获取图片128个特征
def return_128_features(path_img):
    #读图片
    img_rd = io.imread(path_img)
    #把图片转化为灰度图
    img_grey = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
    
    face = detector(img_grey,1)
    print("%-40s %-20s" % ("检测到人脸的图像 / image with faces detected:", path_img), '\n')
    if len(face) != 0:
        shape = predictor(img_grey,face[0])
        face_descriptor = face_rec.compute_face_descriptor(img_grey,shape)
    else:
        face_descriptor = 0
        print('no face')
    return face_descriptor

def return_features_mean_personX(path_faces_personX):
    features_list_personX = []
    picture_list = os.listdir(path_faces_personX)
    if picture_list:
       for i in range(len(picture_list)):
            print("%-40s %-20s" % ("正在读的人脸图像 / image to read:", path_faces_personX + "/" + picture_list[i]))
            feature_128d = return_128_features(path_faces_personX + "/" + picture_list[i])
            if feature_128d == 0:
               i+=1
            else:
                features_list_personX.append(feature_128d)
    else:
        print("文件夹内图像文件为空 / Warning: No images in " + path_faces_personX + '/', '\n')
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean_personX = '0'
    return features_mean_personX

people = os.listdir(path_image_from_camera)
people.sort()

with open("data/features_all.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for person in people:
        print("##### " + person + " #####")
        features_mean_personX = return_features_mean_personX(path_image_from_camera+ "/"+ person)
        writer.writerow(features_mean_personX)
        print("特征均值 / The mean of features:", list(features_mean_personX))
        print('\n')
    print("所有录入人脸数据存入 / Save all the features of faces registered into: data/features_all.csv")

