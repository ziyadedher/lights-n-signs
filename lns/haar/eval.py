import sys 
import cv2
import os
import time
from pathlib import Path

def evaluate(data_path, model_path, num_neighbors=3, scale=1.3):
    """Validate the haar cascade detector loaded from {model_path} using positive samples from {data_path}.

    data_path: corresponds to the positive samples. 
        - Each line in this file corresponds to info for 1 image
            <image path> <number of objects n> x_1 y_1 w_1 h_1 x_2 y_2 w_2 h_2 … x_n y_n w_n h_n
    model_path: the trained haar cascade detector.

    Example usage:
    >>> evaluate('/mnt/ssd1/lns/resources/processed/haar/Y4Signs/annotations/No_Left_Turn_Text_positive',
                 '/mnt/ssd1/lns/resources/trainers/haar/matthieu_haar_y4signs_1/cascade/cascade.xml')
    """

    # Load model and data
    cascade = cv2.CascadeClassifier(model_path)
    with open(data_path) as f:
        images_info = f.readlines()


    total_num_gt = 0
    fp, tp = 0, 0
    for line in images_info:
        # print(line)
        info = line.split(" ")
        img_path, num_signs = info[0], int(info[1])

        path = Path(data_path).parent
        # If the image has actual signs, get their ground-truth coordinates
        if num_signs > 0:
            all_gt = []
            num_gt = int((len(info) - 2) / 4)
            assert num_signs == num_gt, f"The number of ground-truth coordinates and of signs don't match in {img_path}"
            total_num_gt += num_gt

            for i in range(num_gt):
                start_index = 2 + 4*i
                gt_coordinates = (float(info[start_index]), 
                                  float(info[start_index+1]), 
                                  float(info[start_index+2]), 
                                  float(info[start_index+3]))
                all_gt.append(gt_coordinates)
        else:
            continue

        # Get the model's detections
        gray_img = cv2.imread((path/img_path).__str__())
        detections = cascade.detectMultiScale(gray_img, scale, num_neighbors)
        for (x_det, y_det, w_det, h_det) in detections:
            for (x, y, w, h) in all_gt:
                overlap = IOU(x_det, y_det, w_det, h_det, x, y, w, h)
                print(overlap)
                if  overlap> 0.5:
                    tp += 1
                    break
            else:
                fp += 1

    # Report evaluation metrics
    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(total_num_gt)

    print("TP: {}\nFP: {}\nPrecision: {:.2f}\nRecall: {:.2f}\nF1 score: {:.2f}".format(tp, fp, precision, recall, f1_score(precision, recall)))
    return [tp, fp, precision, recall, f1_score(precision, recall)]


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
