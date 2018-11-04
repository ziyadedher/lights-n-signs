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

def find_avg_bb_overlap(true_structure, detection_structure, det_true_map):
    """Find the average overlap of all bounding boxes in data structure
    
    Input: Full data structure for ground truth and detection 
    Output: The average overlap
    """
    def create_bb(annotation: Dict[str, int]) -> List[int]:
        """Take a detection annotation and return a list with bb coords """
        bound_box = []
        bound_box.append(annotation["x_min"])
        bound_box.append(annotation["y_min"])
        bound_box.append(annotation["x_max"])
        bound_box.append(annotation["y_max"])
        return bound_box

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

    total_error = 0
    num_box_overlaps = 0
    for image in detection_structure.keys():
        det_list = detection_structure[image]
        true_list = true_structure[image]

        for det_ind, detection in enumerate(det_list):
            if det_true_map[image][det_ind] is None:
                pass
            else:
                det_box = create_bb(detection)
                true_box = create_bb(true_list[det_true_map[image][det_ind]])
                total_error += compute_bb_overlap(det_box, true_box)
                num_box_overlaps += 1

    return total_error/num_box_overlaps

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

def find_classification_accuracy_stats(confusion_matrix, dataset):
    """Find the recall, precision and f1 score for detections"""

    def find_accuracy(confusion_matrix):
        '''Compute total accuracy of all detections'''
        right_dets = 0
        wrong_dets = 0
        for det_class_name in confusion_matrix.keys():
            for true_class_name in det_class_name.keys():
                if det_class_name == true_class_name:
                    right_dets += confusion_matrix[det_class_name][true_class_name]
                else:
                    wrong_dets += confusion_matrix[det_class_name][true_class_name]
        return right_dets/(right_dets + wrong_dets)

    def find_precision(confusion_matrix, class_name):
        '''Compute precision for given class name'''
        num_det =
        for dets in confusion_matrix[class_name].keys():
            num_det +=


    #accuracy_stats_dict: Dict[str, float] = {}
    # class_acc: Dict[str, Dict[str, int]] = {}
    # for class_name in dataset.classes:
    #     class_acc[class_name] = {}
    #     class_acc[class_name]["right"] = 0
    #     class_acc[class_name]["wrong"] = 0
    # class_acc["None"] = 0
    #
    # #Populate the data structure
    # for image in detection_structure.keys():
    #     det_list = detection_structure[image]
    #     true_list = true_structure[image]
    #
    #     for det_ind, detection in enumerate(det_list):
    #         # handle case where detection without true feature
    #         true_ind = det_true_map[image][det_ind]
    #         detect_type = dataset.classes[detection["class"]]
    #         true_type = dataset.classes[dataset.annotations[image][true_ind]["class"]]
    #
    #         if true_ind is None:
    #             class_acc["None"] += 1
    #         else:
    #             if detect_type == true_type:
    #                 class_acc[true_type]["right"] += 1
    #             else:
    #                 class_acc[true_type]["wrong"] += 1
    #
    # #Compute the precision for each class
    # for class_name in class_acc.keys():
    #     accuracy_stats_dict[class_name] = {}
    #     accuracy_stats_dict[class_name]["accuracy"] = \  #Find adccuracy
    #        class_acc[class_name]["right"]/(class_acc[class_name]["right"] + class_acc[class_name]["wrong"])
    #
    #     for   #find precision
    #     accuracy_stats_dict[class_name]["precision"] =


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
    avg_bb_overlap = find_avg_bb_overlap(dataset.test_annotations,
                                             detection_annotations,
                                             predict_to_truth_map)

    find_classification_accuracy_stats(confusion_matrix)
    print ("Average Bounding Box Overlap: ", avg_bb_overlap)
    print(confusion_matrix)


if __name__ == '__main__':
    benchmark_model("LISA")