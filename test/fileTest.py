import numpy as np
import cv2
import sys
import os
import math
from random import random
from time import time

#This file will process images and show them for validation purposes.

#pass in the cascade path as the first argument and the image directory path as the second

#Will be used to score the cascades from a specified file. Prints out the result
def validateImages(cascade, negFile=None, posFile=None):
    img_cascade = cv2.CascadeClassifier(cascade)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    incorrect = 0
    neg_count = 0
    total_overlap = 0
    pos_count = 0
    correct = 0

    #neighbours represents the number of overlapping bounding boxes required to
    #make a detection. Set to 0 to see ALL detected bounding boxes

    neighbours = 2

    #scale is the rate at which the feature detectors in grown by each interation.
    #Lower scale means more detections but slower runtime.

    scale = 1.2

    try:
        if negFile != None:
            neg = open(negFile, 'r').read()
            neg = neg.replace("\t", " ").split("\n")

            for line in neg:
                if line == "":
                    continue

                path = line.split(" ")[0]
                print(path)
                img = cv2.imread(path, 0)
                img = clahe.apply(img)

                detections = img_cascade.detectMultiScale(img, scale, neighbours)

                for (x,y,w,h) in detections:
                    incorrect += 1

                neg_count += 1
    except KeyboardInterrupt:
        print("Mannually ended this stage")


    try:
        if posFile != None:
            pos = open(posFile, 'r').read()
            pos = pos.replace('\t', ' ').split('\n')

            for line in pos:
                if line == "":
                    continue

                line = line.split(" ")
                path = line[0]
                num = int(line[1])
                print(path)
                img = cv2.imread(path, 0)
                img = clahe.apply(img)

                detections = img_cascade.detectMultiScale(img, scale, neighbours)

                expected = []

                for i in range(num):
                    x,y,w,h = [int(elem) for elem in line[(2+i*4):(6+i*4)]]
                    expected.append([x,y,w,h])
                    pos_count += 1

                for (x2,y2,w2,h2) in detections:
                    for (x1,y1,w1,h1) in expected:
                        prev = total_overlap
                        total_overlap += calculateOverlap([x2,y2,w2,h2], [x1,y1,w1,h1])
                        if total_overlap != prev:
                            correct += 1
    except KeyboardInterrupt:
        print("Mannually ended this stage")

    print("Incorrect detection rate: {}".format(incorrect/neg_count))
    print("Average overlapping dection area: {}".format(total_overlap/pos_count))
    print("Average correct detection rate: {}".format(correct/pos_count))

def calculateOverlap(square1, square2):
    (x1,y1,w1,h1) = square1
    (x2,y2,w2,h2) = square2

    coord1 = np.array([max(x1, x2), max(y1, y2)])
    coord2 = np.array([min(x1+w1, x2+w2), min(y1+h1, y2+h2)])
    size = coord2 - coord1
    return np.prod(size) / (((h1*w1) * (h2*w2)) ** 0.5)


def visualizeImages(cascade, folder):
    img_cascade = cv2.CascadeClassifier(cascade)

    if folder[-1] != '/':
        folder = folder + "/"

    #neighbours represents the number of overlapping bounding boxes required to
    #make a detection. Set to 0 to see ALL detected bounding boxes

    neighbours = 2

    #scale is the rate at which the feature detectors in grown by each interation.
    #Lower scale means more detections but slower runtime.

    scale = 1.2

    files = os.listdir(folder)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

    for f in files:
        img = cv2.imread(folder + f, 1)
        gray = clahe.apply(img)

        detections = img_cascade.detectMultiScale(gray, scale, neighbours)

        for (x,y,w,h) in detections:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            cv2.imshow('img',img)
            k = cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__=="__main__":
    #visualizeImages(sys.argv[1], sys.argv[2])
    validateImages(sys.argv[1], negFile=sys.argv[2], posFile=sys.argv[3])
