from typing import List, Dict, Optional

import math
import cv2

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

def find_classification_accuracy_stats(confusion_matrix, dataset):
    """Find the recall, precision and f1 score for detections"""

    def find_recall(confusion_matrix, class_name):
        num_true = 0
        num_acc = 0
        for det in confusion_matrix.keys():
            num_true += confusion_matrix[det][class_name]
            if det == class_name:
                num_acc = confusion_matrix[det][class_name]

        return float(num_acc/num_true)


    def find_precision(confusion_matrix, class_name):
        '''Compute precision for given class name'''
        num_det = 0
        num_acc = 0
        for g_truth in confusion_matrix[class_name].keys():
            num_det += confusion_matrix[class_name][g_truth]
            if g_truth == class_name:
                num_acc = confusion_matrix[class_name][g_truth]

        return float(num_acc/num_det)

    def find_f1(precision, recall):
        return precision * recall * 2 / (precision + recall)

    for det_class in confusion_matrix.keys():
        precision = find_precision(confusion_matrix, det_class)
        recall = find_recall(confusion_matrix, det_class)
        f1 = find_f1(precision, recall)
        print('For ', class_name, ': precision=', precision,
              ' recall=', recall,' f1=', f1, '\n')


def benchmark_model(dataset: Dataset, model):

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
    print(find_accuracy(confusion_matrix))
    print(find_classification_accuracy_stats(confusion_matrix, dataset))


if __name__ == '__main__':
    from preprocess.preprocessing import preprocess_LISA
    import sys
    # insert the relevant home directory path here #sys.path.append("/Users/RobertAdragna/Documents/AutoDrive/2018-2019/code/lights-n-signs-training/yolov3")
    from yolov3 import yolo
    dataset = preprocess_LISA("LISA")
    model = yolo.YOLO
    benchmark_model(dataset, model)
