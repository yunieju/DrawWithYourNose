import cv2
import numpy as np
import time
import os

# 1. draw with nose tip
# 2. change to draw mode if a mouth is closed
# 3. change to select mode if a mouth is open
folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:

    # 1. Import image
    success, img = cap.read()


    cv2.imshow("Image", img)

    cv2.waitKey(1)