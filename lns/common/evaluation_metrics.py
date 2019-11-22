from typing import Tuple, Dict
from lns.common.model import Model
from lns.common.dataset import Dataset
import itertools
from lns.common.structs import Bounds2D
ConfusionMatrix = Dict[str, Dict[str, int]]

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


def combine_dicts(dict_original:Dict, dict_new:Dict) -> Dict:
    for class_name_outer in dict_new.keys():
        for class_name_inner in dict_new.keys():
            dict_original[class_name_outer][class_name_inner] += dict_new[class_name_outer][class_name_inner]
    return dict_original


def compute_stats(cm, classes):
    TP, FP, FN = 0, 0, 0
    for class_name in classes:
        TP += cm[class_name][class_name]
        FN += sum([cm[class_name][other] for other in classes if other != class_name])
    FP += sum([cm["__none__"][class_name] for class_name in classes])
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
    aggregateConfusionMatrix = {class_name: {class_name: 0 for class_name in classes} for class_name in classes}
    annotations = dataset.annotations
    image_paths = annotations.keys()  # returns images paths in the data set
    IOU_aggregate = 0.0  # only for TP?
    count = 0
    TP_aggregate, FP_aggregate, FN_aggregate = 0, 0, 0

    for image in image_paths:
        target_obj = annotations[image]
        predictions = model.predict_path(image)
        boolean_detected = [1]*len(target_obj)  # identify gt objects that have been detected by model
        boolean_matched = [1]*len(predictions)  # identify predictions that correspond to some gt
        confusionMatrix = {class_name: {class_name: 0 for class_name in classes} for class_name in classes}

        for j, pred_obj in enumerate(predictions):
            confusionMatrix = {class_name: {class_name: 0 for class_name in classes} for class_name in classes}
            for i, ground_truth_obj in enumerate(target_obj):
                if boolean_detected[i]:
                    # handles duplicate bounding boxes...only check ground truth object if it hasn't been matched
                    if iou(pred_obj.bounds, ground_truth_obj.bounds) >= threshold:
                        if pred_obj.class_index == ground_truth_obj.class_index:
                            # really good prediction... TP
                            confusionMatrix[classes[ground_truth_obj.class_index]][classes[pred_obj.class_index]] += 1
                            boolean_detected[i] = 0  # mark as detected on both
                            boolean_matched[j] = 0
                            IOU_aggregate +=iou(pred_obj.bounds, ground_truth_obj.bounds)
                            count += 1
                        else:
                            # good IOU but incorrect label...FN
                            confusionMatrix[classes[ground_truth_obj.class_index]][classes[pred_obj.class_index]] += 1
                            boolean_detected[i] = 0  # mark as detected on both
                            boolean_matched[j] = 0
                    # elif iou(pred_obj.bounds, ground_truth_obj.bounds) < threshold:
                    #     if pred_obj.class_index == ground_truth_obj.class_index:
                    #         # predictions that have a correct predicted label but not enough IoU
                    #         # FP
                    #         continue
                    #     else:
                    #         # no correspondence to ground truth
                    #         continue

            # apply boolean masks to extract objects that were not detected or not matched with a ground truth object
            undetected_gt = itertools.compress(target_obj, boolean_detected)
            for obj in undetected_gt:  # FN
                confusionMatrix[classes[obj.class_index]]["__none__"] += 1

            unmatched_pred = itertools.compress(predictions, boolean_matched)
            for obj in unmatched_pred:  # FP
                confusionMatrix["__none__"][classes[obj.class_index]] += 1

        TP, FP, FN = compute_stats(confusionMatrix, classes)
        TP_aggregate += TP
        FP_aggregate += FP
        FN_aggregate += FN

        # add CM of this image to CM of the entire data set
        aggregateConfusionMatrix = combine_dicts(aggregateConfusionMatrix, confusionMatrix)

    if TP_aggregate+FP_aggregate:
        stats["precision"] = TP_aggregate / (TP_aggregate + FP_aggregate)
    if TP_aggregate+FN_aggregate:
        stats["recall"] = TP_aggregate / (TP_aggregate + FN_aggregate)
    if 2*TP_aggregate+FP_aggregate+FN_aggregate:
        stats["f1"] = 2 * (stats["precision"]) * (stats["recall"]) / (stats["precision"] + stats["recall"])

    return IOU_aggregate/count, aggregateConfusionMatrix, stats
