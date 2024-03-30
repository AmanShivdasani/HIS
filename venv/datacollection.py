import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
folder="data/Z"
counter = 0

while True:
  success,img=cap.read()
  hands,img=detector.findHands(img)
  if hands:
    hand=hands[0]
    x,y,w,h = hand['bbox']
    # imgwhite = np.ones((imgSize,imgSize,3),np.unit8)
    # imgcrop = img[y:y+h, x:x+w]
    # cv2.imshow("ImageCrop",imgcrop)
  cv2.imshow("Image",img)
  key=cv2.waitKey(1)
  if key == ord("s"):
    counter+=1
    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',img)
    print(counter)
  