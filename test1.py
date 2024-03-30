import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
# import time

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=2)
# classifier = Classifier("model/keras_model.h5","model/labels.txt")
labels=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
offset=20
imgsize=300
while True:
  success , img=cap.read()
  hands,img=detector.findHands(img)
  if hands:
    hand=hands[0]
    x , y, w , h = hand["bbox"]
  cv2.imshow("Img",img)
  key=cv2.waitKey(1)
  
  # 
  