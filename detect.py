import dlib         # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 Numpy
import cv2          # 图像处理的库 OpenCv

from PIL import Image 
import os           # 读写文件
import shutil       # 读写文
# Dlib 正向人脸检测器 / frontal face detector
detector = dlib.get_frontal_face_detector()
# Dlib 68 点特征预测器 / 68 points features predictor
predictor = dlib.shape_predictor('/Users/xitongzhou/Desktop/faceDetect/data/dlib/shape_predictor_68_face_landmarks.dat')
img = cv2.imread('test.jpeg')
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
faces = detector(img_gray, 0)
if len(faces) != 0: 
    for d in faces:
        color_rectangle = (255, 255, 255)
        cv2.rectangle(img,tuple([d.left(), d.top()]),tuple([d.right() , d.bottom()]),color_rectangle, 2)
        cv2.putText(img, "Faces: " + str(len(faces)), (20, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        

cv2.namedWindow("image") #创建窗口并显示的是图像类型
cv2.imshow("image",img)
cv2.waitKey(0)        #等待事件触发，参数0表示永久等待
cv2.destroyAllWindows()