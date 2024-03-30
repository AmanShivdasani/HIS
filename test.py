import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from cvzone.ClassificationModule import Classifier
import time
import math

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
offset=20
classifier = Classifier("C:\signlangu\model\keras_model.h5","C:\signlangu\model\labels.txt")
labels=["A","B"]
folder="data/B"
counter = 0
imgsize=300
while True:
  success,img=cap.read()
  imgOutput=img.copy()
  hands,img=detector.findHands(img)
  if hands:
    hand=hands[0]
    x,y,w,h = hand['bbox']
    imgWhite=np.ones((imgsize,imgsize,3),np.uint8)*255
    imgCrop=img[y-offset:y +h+offset,x-offset:x +w+offset]
    imgCropShape=imgCrop.shape
    aspectRatio=h/w
    prediction,index=classifier.getprediction(img)
    if aspectRatio > 1:
      k=imgsize/h
      wCal=math.ceil(k*w)
      imgResize=cv2.resize(imgCrop,(wCal,imgsize))
      imgResizeShape=imgResize.shapewGap=math.ceil((imgsize-wCal)/2)
      wGap=math.ceil((imgsize-wCal)/2)
      imgWhite[:,wGap:wCal+wGap]=imgResize
      prediction,index=classifier.getprediction(img)
      print(prediction,index)
    else:
      k=imgsize/w
      hCal=math.ceil(k*h)
      imgResize=cv2.resize(imgCrop,(imgsize,hCal))
      imgResizeShape=imgResize.shape
      hGap=math.ceil((imgsize-hCal)/2)
      imgWhite[hGap:hCal+hGap,:]=imgResize
      prediction,index=classifier.getPrediction(img)
    cv2.putText(img,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,0,255),2)
    cv2.rectangle(imgOutput,(x-offset,y-offset),(y+w+offset,y+h+offset),(255,0,255),4)
    cv2.imshow("ImageCrop",imgCrop)
    cv2.imshow("imgwhite",imgWhite)
    cv2.waitKey(2)
  cv2.imshow("Image",imgOutput)
  cv2.waitKey(1)
  