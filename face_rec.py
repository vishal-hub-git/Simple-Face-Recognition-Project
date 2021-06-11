# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:42:47 2020

@author: balaji
"""

import cv2 as cv
import numpy as np
import os
from skimage import io
features=[]
labels=[]
def train():
    p=[]
    p1=[]
    p2=[]
    for i in os.listdir('Dhoni'):
        p.append(i)
    for j in os.listdir('Kohli'):
        p1.append(j)
    for k in os.listdir('Rohit'):
        p2.append(k)
    label=0
    for i in p:
        img=os.path.join('Dhoni',i)     
        img_arr=io.imread(img)
        gray=cv.cvtColor(img_arr,cv.COLOR_BGR2GRAY)
        haar=cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        faces=haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=15)
        for (x,y,w,h) in faces:
            faces_roi=gray[y:y+h,x:x+h]
            features.append(faces_roi)
            labels.append(label)
    label=1
    for j in p1:
        img=os.path.join('Kohli',j)
        img_arr=io.imread(img)
        gray=cv.cvtColor(img_arr,cv.COLOR_BGR2GRAY)
        haar=cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        faces=haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=15)
        for (x,y,w,h) in faces:
            faces_roi=gray[y:y+h,x:x+w]
            features.append(faces_roi)
            labels.append(label)
    label=2
    for k in p2:
        img=os.path.join('Rohit',k)
        img_arr=io.imread(img)
        gray=cv.cvtColor(img_arr,cv.COLOR_BGR2GRAY)
        haar=cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        faces=haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=15)
        for (x,y,w,h) in faces:
            faces_roi=gray[y:y+h,x:x+w]
            features.append(faces_roi)
            labels.append(label)
train()
features=np.array(features,dtype='object')
labels=np.array(labels)
rec=cv.face.LBPHFaceRecognizer_create()
print(len(labels),len(features))
rec.train(features,labels)
np.save('features.npy',features)
rec.save('rec.yml')


