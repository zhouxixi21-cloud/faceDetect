
import dlib         # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 Numpy
import cv2          # 图像处理的库 OpenCv

import os           # 读写文件
import shutil       # 读写文件

cap = cv2.VideoCapture(0)

# 设置视频参数 set camera
cap.set(3, 480)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')
while cap.isOpened():
    # 480 height * 640 width
    flag, img_rd = cap.read()
    kk = cv2.waitKey(1)

    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    kk = cv2.waitKey(1)
    faces = detector(img_gray, 0)
    font = cv2.FONT_HERSHEY_COMPLEX

    if len(faces) != 0:
        for k , d  in enumerate(faces):
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())

            hh = int(height/2)
            ww = int(width/2)
                
            # 设置颜色 / the color of rectangle of faces detected
            color_rectangle = (255, 255, 255)
                

            cv2.rectangle(img_rd,
                            tuple([d.left() - ww, d.top() - hh]),
                            tuple([d.right() + ww, d.bottom() + hh]),
                            color_rectangle, 2)   

    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    if kk == ord('q'):
        break
    cv2.imshow("camera", img_rd)
cap.release()

cv2.destroyAllWindows()