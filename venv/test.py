import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from cvzone.ClassificationModule import Classifier
import time

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
classifier = Classifier("model/keras_mode.h5","model/labels.txt")
labels=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
# folder="data/Z"
# counter = 0

while True:
  success,img=cap.read()
  hands,img=detector.findHands(img)
  prediction,index=classifier.getPrediction(img)
  print(prediction,index)
  if hands:
    hand=hands[0]
    x,y,w,h = hand['bbox']
    # imgwhite = np.ones((imgSize,imgSize,3),np.unit8)
    # imgcrop = img[y:y+h, x:x+w]
    # cv2.imshow("ImageCrop",imgcrop)
  cv2.imshow("Image",img)
  cv2.waitKey(1)
  