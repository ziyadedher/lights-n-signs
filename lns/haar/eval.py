import sys 
import cv2
import os
import time
import numpy as np
from pathlib import Path

def evaluate(data_path, model_path, trainer_path, num_neighbors=3, scale=1.1):
    """Validate the haar cascade detector loaded from {model_path} using positive samples from {data_path}.

    data_path: corresponds to the positive samples. 
        - Each line in this file corresponds to info for 1 image
            <image path> <number of objects n> x_1 y_1 w_1 h_1 x_2 y_2 w_2 h_2 … x_n y_n w_n h_n
    model_path: the trained haar cascade detector. xml file
    trainer_path: path to the trainer folder

    Example usage:
    >>> evaluate('/mnt/ssd1/lns/resources/processed/haar/Y4Signs/annotations/No_Left_Turn_Text_positive',
                 '/mnt/ssd1/lns/resources/trainers/haar/matthieu_haar_y4signs_1/cascade/cascade.xml')
    """

    # Load model and data
    cascade = cv2.CascadeClassifier(model_path)
    with open(data_path) as f:
        images_info = f.readlines()
    
    # Determine name of folder where to save results
    to_save = os.path.join(trainer_path, f'visual_{num_neighbors}_{scale}_0')
    index_ = 1
    while os.path.exists(to_save):
        to_save = os.path.join(trainer_path, f'visual_{num_neighbors}_{scale}_{index_}')
        index_ += 1
        
    # Prepare folders to save images
    os.mkdir(to_save)

    total_num_gt = 0
    fp, tp, fn = 0, 0, 0
    for line in images_info:
        info = line.split(" ")
        img_path, num_signs = info[0], int(info[1])

        # If the image has actual signs, get their ground-truth coordinates
        if num_signs > 0:
            all_gt = []
            num_gt = int((len(info) - 2) / 4)
            assert num_signs == num_gt, f"The number of ground-truth coordinates and of signs don't match in {img_path}"
            total_num_gt += num_gt

            for i in range(num_gt):
                start_index = 2 + 4*i
                xmin = info[start_index]
                ymin = info[start_index+1]
                width = info[start_index+2]
                height = info[start_index+3]
                gt_coordinates = (int(xmin), 
                                  int(ymin), 
                                  int(width), 
                                  int(height))
                all_gt.append(gt_coordinates)
        else:
            continue


        annotations_path = Path(data_path).parent
        gray_img = cv2.imread(str(annotations_path/img_path), 0)
        out_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        im_name = img_path[str(img_path).rindex('/') + 1:]
        
        # Get the model's detections
        detections = cascade.detectMultiScale(gray_img, scale, num_neighbors)
        img_path_save = os.path.join(to_save, im_name)

        # Find TP and FN
        for (x, y, w, h) in all_gt:
            for (x_det, y_det, w_det, h_det) in detections:
                iou = IOU(x_det, y_det, w_det, h_det, x, y, w, h)
                if iou > 0.5:
                    tp += 1
                    break 
                    # Once we find one prediction sufficiently overlapping with the gt, stop. 
                    # This is to prevent TP from inflating due to duplicates (when min_neighbours is low).
            else:
                fn += 1
        
        # Find FP
        for (x_det, y_det, w_det, h_det) in detections:
            for (x, y, w, h) in all_gt:
                iou = IOU(x_det, y_det, w_det, h_det, x, y, w, h)
                if iou > 0.5:
                    break
            else:
                fp += 1
        
        # Find maximum IOU TODO: integrate with TP, FN so get rid of redundant IOU calls
        all_ious = []
        for (x, y, w, h) in all_gt:
            best_det_iou = 0
            for (x_det, y_det, w_det, h_det) in detections:
                iou = IOU(x_det, y_det, w_det, h_det, x, y, w, h)
                if iou > best_det_iou:
                    best_det_iou = iou
            all_ious.append(best_det_iou)
        avg_iou = np.mean(all_ious)

        # Draw bounding boxes
        for (x, y, w, h) in all_gt:
            # GT bounding box -> red
            cv2.rectangle(out_img, (x, y), (x+w, y+h), (0, 0, 255), 2) 
            for (x_det, y_det, w_det, h_det) in detections:
                # Pred bounding box -> cyan
                cv2.rectangle(out_img, (x_det, y_det), (x_det+w_det, y_det+h_det), (255, 255, 0), 2) 
        
        cv2.imwrite(img_path_save, out_img)

    # Report evaluation metrics
    try:
        precision = float(tp) / float(tp + fp)
        recall = float(tp) / float(tp + fn)
        f1 = f1_score(precision, recall)
    except ZeroDivisionError as e:
        print('No bounding boxes were detected. Try decreasing num_neighbours or scale_factor. There might be a bug in the code as well.')

    msg = f"TP: {tp}\nFP: {fp}\nFN: {fn}\nPrecision: {precision}\nRecall: {recall}\nF1 score: {f1}\nIoU: {avg_iou}"
    print(msg)

    
    file = open(os.path.join(trainer_path, f'results_{num_neighbors}_{scale}.txt'), "w")
    file.write(msg)
    file.close()

    return [tp, fp, precision, recall, f1]


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
