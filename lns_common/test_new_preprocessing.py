from typing import List, Dict, Optional

import math
import cv2

from lns_common.model import Model
from lns_common.preprocess.preprocessing import Dataset


def get_img_dist(image_1: List[int], image_2: List[int]) -> float:
    """Find the euclidean distance between two images measured from each corner
    Inputs: image_1,2 - list of [x_min, y_min, x_max. y_max] for each image
    Outputs: Distance value """
    if len(image_1) != len(image_2):
        return -1

    sum = 0
    for i in range(0, len(image_1)):
        sum += math.pow(abs(image_2[i] - image_1[i]), 2)
    return float(math.sqrt(sum))


def compute_bb_overlap(detection: List[int], g_truth: List[int]) -> float:
    """Find the area overlap between two specified rectangles.
    Inputs: list of [x_min, y_min, x_max. y_max]
    Outputs: Fraction of overlapping bounding boxes"""

    def find_overlap(image_1: List[int], image_2: List[int]) -> int:
        dim = [x for x in image_1 if x in image_2]
        return len(dim)

    if len(detection) != len(g_truth):
        return -1

    overlap_width = find_overlap(list(range(detection[0], detection[2] + 1)),
                                 list(range(g_truth[0], g_truth[2] + 1)))
    overlap_height = find_overlap(list(range(detection[1], detection[3] + 1)),
                                  list(range(g_truth[1], g_truth[3] + 1)))
    area_g_truth = (g_truth[2] - g_truth[0]) * (g_truth[3] - g_truth[1])
    return float((overlap_height * overlap_width) / area_g_truth)


def find_confusion_matrix(true_structure, detection_structure, det_true_map,
                          dataset):
    '''Take a full test results data structure and divide into true positive,
     true negatives, etc'''

    # Init conf_matrix. Every predicted-actual combo is single int entry
    confusion_matrix: Dict[str, Dict[str, int]] = {}
    class_names = dataset.classes
    class_names.append("None")
    for row_type in class_names:
        confusion_matrix[row_type] = {}
        for col_type in class_names:
            confusion_matrix[row_type][col_type] = 0

    # populate the matrix
    for image in detection_structure.keys():
        det_list = detection_structure[image]
        true_list = true_structure[image]
        detected_true_indices = []  # the true indices that map to detections

        for det_ind, detection in enumerate(det_list):
            # handle case where detection without true feature
            if det_true_map[image][det_ind] is None:
                confusion_matrix[detection["class"]]["None"] += 1
            else:
                confusion_matrix[detection["class"]][dataset.classes[
                    true_list[det_true_map[image][det_ind]]["class"]]] += 1

            detected_true_indices.append(det_ind)

        # handle case where true feature without detection
        if len(true_list) > len(det_list):
            undetected_true_indices = [x for x in range(0, len(true_list))
                                       if x not in detected_true_indices]
            for index in undetected_true_indices:
                confusion_matrix["None"][dataset.classes[index]] += 1

    return confusion_matrix


def benchmark_model(dataset: Dataset, model: Optional[Model]):

    # Unpack the images
    detection_annotations: Dict[str, List[Dict[str, int]]] = {}
    predict_to_truth_map: Dict[str, Dict[int, int]] = {}
    print(len(dataset.test_annotations))
    for img_path in dataset.test_annotations:
        # Get model's detections
        img_file = cv2.imread(img_path)
        detections_list = model.predict(img_file)
        print(img_path)

        # Iterate through detections, map each to the closest ground truth
        predict_to_truth_map[img_path] = {}
        for ind, det in enumerate(detections_list):
            if (len(dataset.test_annotations[img_path])
                    - len(predict_to_truth_map[img_path])) > 0:
                closest = [-1, -1]  # Elem0=distance, elem1=closest val ind
                for index, tru_li in enumerate(
                        dataset.test_annotations[img_path]):
                    feature_dist = get_img_dist([tru_li["x_min"],
                                                 tru_li["y_min"],
                                                 tru_li["x_max"],
                                                 tru_li["y_max"]],
                                                [det.bounding_box.left,
                                                 det.bounding_box.top,
                                                 det.bounding_box.left
                                                 + det.bounding_box.width,
                                                 det.bounding_box.top
                                                 + det.bounding_box.height])
                    if ((closest[1] == -1) or (feature_dist < closest[0])) \
                            and (index not in
                                 predict_to_truth_map[img_path].values()):
                        closest[0] = feature_dist
                        closest[1] = index
                predict_to_truth_map[img_path][ind] = closest[1]
            else:
                predict_to_truth_map[img_path][ind] = None

        # Populate test_annotations-like structure of detections
        detection_annotations[img_path] = []
        for index, detection in enumerate(detections_list):
            detection_annotations[img_path].append({})

            detection_annotations[img_path][index]["class"] = \
                detection.predicted_classes[0]
            detection_annotations[img_path][index]["x_min"] = \
                detection.bounding_box.left
            detection_annotations[img_path][index]["x_max"] = \
                detection.bounding_box.left + detection.bounding_box.width
            detection_annotations[img_path][index]["y_min"] = \
                detection.bounding_box.top
            detection_annotations[img_path][index]["y_max"] = \
                detection.bounding_box.top + detection.bounding_box.height

    # Generate and print all accuracy statistics
    confusion_matrix = find_confusion_matrix(dataset.test_annotations,
                                             detection_annotations,
                                             predict_to_truth_map, dataset)
    print(confusion_matrix)


if __name__ == '__main__':
    benchmark_model("LISA")