from typing import Tuple, Dict
from lns.common.model import Model
from lns.common.dataset import Dataset
import cProfile
import tqdm
from lns.common.structs import Bounds2D
import numpy as np
ConfusionMatrix = Dict[str, Dict[str, int]]
#Hungarian Algorithm for association between label and prediction

def iou(box1: Bounds2D, box2: Bounds2D) -> float:
    """Calculate the intersection-over-union of two bounding boxes."""
    x_left = max(box1.left, box2.left)
    y_top = max(box1.top, box2.top)
    x_right = min(box1.right, box2.right)
    y_bottom = min(box1.bottom, box2.bottom)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area / float(box1.area + box2.area - intersection_area)

def compute_stats(cm, classes):
    TP, FP, FN = 0, 0, 0
    for class_name in range(classes):
        TP += cm[class_name,class_name]
        FN += sum([cm[class_name,other] for other in range(classes) if other != class_name])
        FP += cm[-1,class_name]
    #FP you detect something but the truth value is none (does not exist)
    #FN is literally everything else
    #TP is along the diagnol
    return TP, FP, FN


def benchmark(model: Model, dataset: Dataset, threshold: float):
    # INPUTS:
    # model is the model used to compute predictions
    # data set is the structure that contains classes, annotations and the data

    # using the concept of "nothingness"
    # Get the classes and add the `none` class for predictions that are less than the IOU threshold
    # Create a confusion matrix where the keys of the outer dict are the row names (true classes)
    # the keys of the inner dict are the column names (predicted classes)
    classes = dataset.classes + ["__none__"]
    stats = {stat: 0.0 for stat in ['precision', 'recall', 'f1']}
    aggregateConfusionMatrix = np.zeros((len(classes),len(classes)))
    IOU_aggregate = 0.0  # only for TP?
    count = 0
    annotations = dataset.annotations.items()
    TP_aggregate, FP_aggregate, FN_aggregate = 0, 0, 0
    for image, target_obj in tqdm.tqdm(list(annotations)[:100]):
        predictions = model.predict_path(image)
        detected = np.array([False]*len(target_obj))  # identify gt objects that have been detected by model
        matched = np.array([False]*len(predictions))  # identify predictions that correspond to some gt

        for j, pred_obj in enumerate(predictions):
            if not matched[j]:
                for i, ground_truth_obj in enumerate(target_obj):
                    if not detected[i]:
                            # handles duplicate bounding boxes...only check ground truth object if it hasn't been matched
                            IOU = iou(pred_obj.bounds, ground_truth_obj.bounds) 
                            if IOU >= threshold:
                                if pred_obj.class_index == ground_truth_obj.class_index:
                                    # really good prediction... TP vs. good IOU but incorrect label...FN
                                    IOU_aggregate += IOU
                                    count += 1
                                
                                aggregateConfusionMatrix[ground_truth_obj.class_index, pred_obj.class_index] += 1
                                detected[i], matched[j] = True, True  # mark as detected on both

    # apply boolean masks to extract objects that were not detected or not matched with a ground truth object
    for i in range(len(detected)):
        if not detected[i]:
            aggregateConfusionMatrix[target_obj[i].class_index,-1] += 1
    for i in range(len(matched)):
        if not matched[i]:
            aggregateConfusionMatrix[-1,predictions[i].class_index] += 1

    TP, FP, FN = compute_stats(aggregateConfusionMatrix, len(classes))
    TP_aggregate += TP
    FP_aggregate += FP
    FN_aggregate += FN

    if TP_aggregate+FP_aggregate:
        stats["precision"] = TP_aggregate / (TP_aggregate + FP_aggregate)
    if TP_aggregate+FN_aggregate:
        stats["recall"] = TP_aggregate / (TP_aggregate + FN_aggregate)
    if 2*TP_aggregate+FP_aggregate+FN_aggregate:
        stats["f1"] = 2 * (stats["precision"]) * (stats["recall"]) / (stats["precision"] + stats["recall"])
    return IOU_aggregate/count, aggregateConfusionMatrix, stats
