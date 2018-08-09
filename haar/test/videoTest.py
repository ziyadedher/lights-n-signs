import numpy as np
import cv2
import sys
import os
import math
from random import random
from time import time

def video_test(path):
    watch_cascade = cv2.CascadeClassifier(path)

    #neighbours represents the number of overlapping bounding boxes required to
    #make a detection. Set to 0 to see ALL detected bounding boxes

    neighbours = 2

    #scale is the rate at which the feature detectors in grown by each interation.
    #Lower scale means more detections but slower runtime.

    scale = 1.2

    cap = cv2.VideoCapture(0)
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Applies histogram and greying of picture in this step
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        watches = watch_cascade.detectMultiScale(gray, scale, neighbours)

        for (x,y,w,h) in watches:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_test("../haar/data/cascade.xml")
