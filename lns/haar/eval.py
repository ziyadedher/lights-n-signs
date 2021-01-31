import sys 
import cv2
import os
import time


def evaluate(data_path, model_path, num_neighbors=4, scale=1.3):
    """Validate the haar cascade detector loaded from {model_path} using positive samples from {data_path}.

    data_path: corresponds to the positive samples. 
        - Each line in this file corresponds to info for 1 image
            <image path> <number of objects n> x_1 y_1 w_1 h_1 x_2 y_2 w_2 h_2 … x_n y_n w_n h_n
    model_path: the trained haar cascade detector.

    Example usage:
    >>> evaluate('Y4Signs/annotations/No_Left_Turn_Text_positive','models/left_arrow/cascade_24_large.xml')
    """
    cascade = cv2.CascadeClassifier(model_path)

    with open(data_path) as f:
        images_info = f.readlines()

    fpr, tpr = 0, 0  # True Positive rate & False Positive rate
    for line in images_info:
        info = line.split(" ")
        img_path, num_detected = info[0], int(info[1])

        if num_detected > 0:  # image has detected signs, take the 1st coordinate
            x, y, w, h = int(info[2]), int(info[3]), int(info[4]), int(info[5])
        else:
            continue

        gray_img = cv2.imread(img_path)
        detections = cascade.detectMultiScale(gray, scale, num_neighbors)

        for (x_det, y_det, w_det, h_det) in detections:
            if IOU(x_det, y_det, w_det, h_det, x, y, w, h) > 0.5:
                tpr += 1
            else:
                fpr += 1

    # Report evaluation metrics
    precision = float(tpr) / float(tpr + fpr)
    recall = float(tpr) / float(num_valid)

    print("TPR: {}\nFPR: {}\nPRECISION: {:.2f}\nRECALL: {:.2f}\nF1 SCORE: {:.2f}".format(tpr, fpr, precision, recall, f1_score(precision, recall)))
    return [tpr,fpr,precision,recall,f1_score(precision, recall)]


def IOU(x, y, w, h, x1t, y1t, wt, ht):
    """Return the intersection over union metrics of two bounding boxes.

    IoU is known to be a good metric for measuring overlap between two bounding boxes.
    """
    x2 = x + w
    y2 = y + h
    x2t = x1t + wt
    y2t = y1t + ht

    # Check if the BBs intersect first
    if x2 < x1t or x > x2t or y2 < y1t or y > y2t:
        return 0.0

    xa = max(x, x1t)
    ya = max(y, y1t)
    xb = min(x2, x2t)
    yb = min(y2, y2t)
    intersection = (xb - xa + 1) * (yb - ya + 1)
    area = (w + 1) * (h + 1)
    areat = (wt + 1) * (ht + 1)
    union = area + areat - intersection
    return float(intersection) / float(union)


def f1_score(p, r):
    """Calculate the F1-Score given precision and recall.

    F1-Score is a measure combining both precision and recall, generally described as the harmonic mean of the two. 
    """
    return 0 if (p + r) == 0 else 2 * (p * r)/(p + r)
