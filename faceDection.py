import cv2
import numpy as np
face_pattern = cv2.CascadeClassifier('trainner/trainner.yml')
camera=cv2.VideoCapture(0)
while camera.isOpened:
    (ok, sample_image) = camera.read()
    if not ok:
        break
    faces = face_pattern.detectMultiScale(sample_image,scaleFactor=1.1,minNeighbors=5,minSize=(80,80))
    for (x,y,w,h) in faces:
        cv2.rectangle(sample_image,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('face',sample_image)
    if cv2.waitKey(10)&0xFF==ord('q'):
        break
camera.release()
cv2.destroyAllWindows()