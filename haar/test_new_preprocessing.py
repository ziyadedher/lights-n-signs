import os
import math
import copy
import cv2
from typing import List, Dict

from common import config
from common.model import Bounds2D, PredictedObject2D
from haar.train import Trainer
from haar.model import *
from common import preprocess

def get_image_distance(image_1: List[int], image_2: List[int]) -> float:
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
    """Find the area overlap between two specified rectangles 
    Inputs: image_1,2 - list of [x_min, y_min, x_max. y_max] for each image
    Outputs: Fraction of detection bounding box that overlaps with g_truth's"""

    def find_overlap(image_1: List[int], image_2: List[int]) -> int :
        dim = [x for x in image_1 if x in image_2]
        return len(dim)

    if len(detection) != len(g_truth):
        return -1

    overlap_width = find_overlap(list(range(detection[0], detection[2]+1)),  list(range(g_truth[0], g_truth[2]+1)))
    overlap_height = find_overlap(list(range(detection[1], detection[3]+1)),  list(range(g_truth[1], g_truth[3]+1)))
    area_g_truth = (g_truth[2]-g_truth[0])*(g_truth[3]-g_truth[1])
    return float((overlap_height*overlap_width)/area_g_truth)

def find_average(full_accuracy_stats, param_name):
    '''Take a full test results data structure and find the avg of specified param name (two supported rn)
        Inputs: full_accuracy_stats = Dict[str: Dict[int: Dict[str:union(int, List[int, int, int, int])]]]
        Inputs: param_name - the parameter whose average we want to take
        Inputs: ground_truth - a dict of annotations as specified in test.annotations
        Outputs: float specifiying the resultant average'''

    sum_of_param = 0
    for image_path in full_accuracy_stats.keys() :
        for detect_num in full_accuracy_stats[image_path].keys() :
            if (param_name == "average_type_error") :
                sum_of_param += full_accuracy_stats[image_path][detect_num]["correct_class"]

            elif (param_name == "average_bounding_box_overlap") :
                sum_of_param += full_accuracy_stats[image_path][detect_num]["bounding_box_overlap"]

    return sum_of_param/len(full_accuracy_stats.keys())

def find_confusion_matrix(true_structure, detection_strucutre, true_det_map ):
    '''Take a full test results data structure and divide into true positive, true negatives, etc'''
    pass



class DummyModel(Model):
    def predict(self, img):
        return [PredictedObject2D(Bounds2D(0,0,100,100), ["go"])]


def benchmark_model(dataset_name: str):
    # Setup the model

    dataset = preprocess.preprocess.Preprocessor.preprocess(dataset_name)  #The Dataset object for some specified name
    # trainer = Trainer("trainer", dataset)
    # trainer.setup_training(24, 5000, "go")
    # trainer.train(100, 4000, 2000)
    #model = trainer.generate_model()
    model = DummyModel()

    # Unpack the images
    img_accuracy_stats: Dict[str: Dict[int: Dict[str:union(int, List[int, int, int, int])]]] = {}
    detection_annotations: Dict[str: List[Dict[str,int]]] = {}
    for image_path in dataset.test_annotations:
        # Get model's detections
        img_file = cv2.imread(image_path)
        detections_list = model.predict(img_file)

        # Go through all detections, map each to the closest ground truth detectable object
        predict_to_truth_map = {}
        possible_gt_list = copy.deepcopy(dataset.test_annotations[image_path])
        removal_indicies = []  # allows accurate transform to g_truth indicides


        for ind, detection in enumerate(detections_list):
            if len(possible_gt_list) > 0:
                closest = [-1,-1]  #Element 0 is distance, element 1 is index of closest value
                for index, true_light in enumerate(possible_gt_list):
                    feature_dist = get_image_distance([true_light["x_min"],true_light["y_min"],true_light["x_max"],true_light["y_max"]],
                                       [detection.bounding_box.left, detection.bounding_box.top,
                                       detection.bounding_box.left + detection.bounding_box.width,
                                       detection.bounding_box.top + detection.bounding_box.height])
                    if (closest[1] == -1) or (feature_dist < closest[0]) :
                        closest[0] = feature_dist
                        closest[1] = index

                #Get true index in test_annotations and remove this feature from g_truhts we consider
                del possible_gt_list[closest[1]]

                true_index = 0
                if len(removal_indicies) == 0:
                    true_index = closest[1]
                    removal_indicies.append(closest[1])
                else :
                    true_index = 0
                    for removed in removal_indicies:
                        if closest[1] >= removed :
                            true_index += 1
                    true_index = closest[1] + true_index

                predict_to_truth_map[ind] = true_index

            else:
                predict_to_truth_map[ind] = None


        #Populate data strucutre containing predicted features on each image. Equivilant to test_annotations structure
        detection_annotations[image_path] = []
        for index, detection in enumerate(detections_list):
            detection_annotations[image_path].append({})

            detection_annotations[image_path][index]["class"] = detection.predicted_classes[0]
            detection_annotations[image_path][index]["x_min"] = detection.bounding_box[0]
            detection_annotations[image_path][index]["x_max"] = detection.bounding_box[1]
            detection_annotations[image_path][index]["y_min"] = detection.bounding_box[2]
            detection_annotations[image_path][index]["y_max"] = detection.bounding_box[3]

    #Generate and print all accuracy statistics
    find_confusion_matrix(dataset.test_annotations, detection_annotations, predict_to_truth_map)

if __name__ == '__main__':
    benchmark_model("LISA")
