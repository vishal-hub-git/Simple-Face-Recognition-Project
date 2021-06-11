# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:45:15 2020

@author: balaji
"""

import cv2 as cv

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')

people = ['Dhoni','Kohli','Rohit']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('rec.yml')

img = cv.imread('rt2.jpg')
img=cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 8)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'He is {people[label]}. It is predicted with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
cv.destroyAllWindows()