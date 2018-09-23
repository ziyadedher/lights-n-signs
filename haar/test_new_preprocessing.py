import os
import math
import copy
import cv2
from typing import List

from common import config
from common import model
from haar import train
from haar import model
from common import preprocess

def get_image_distance(image_1: List[int], image_2: List[int]) -> float:
    """Find the euclidean distance between two images measured from each corner
    Inputs: image_1,2 - list of [x_min, y_min, x_max. y_max] for each image
    Outputs: Distance value """
    if len(image_1) != len(image_2):
        return -1

    sum = 0
    for i in range(0, len(image_1)):
        sum += (image_2[i] - image_1[i]) ^ 2
    return float(math.sqrt(sum))

def compute_bb_overlap(image_1: List[int], image_2: List[int]) -> float:
    """Find the area overlap between two specified rectangles 
    Inputs: image_1,2 - list of [x_min, y_min, x_max. y_max] for each image
    Outputs: Overlap value"""
    if len(image_1) != len(image_2):
        return -1

    overlap_width = 4


def benchmark_model(dataset_name: str):
    # Setup the model

    dataset = preprocess.Preprocessor.preprocess(dataset_name)  #The Dataset object for some specified name
    trainer = train.Trainer("trainer", dataset)
    trainer.setup_training(24, 500, "go")
    trainer.train(2, 150, 100)
    model = trainer.generate_model()

    # Unpack the images

    true_image_list = test_info["images"]
    true_classes_list = test_info["classes"]
    true_annotations_dict = test_info["annotations"]  ##



    for image_path in dataset.annotations_test.keys():
        # Get model's detections
        img_file = cv2.imread(image_path)
        detections_list = model.predict(img_file)

        # Check if right # of predictions
        if len(detections_list) == len(dataset.annotations_test[image_path]):
            right_predict_num = True
        else:
            right_predict_num = False

        # Assume the right number of predictions, map ground truth to prediction
        truth_to_predict_map = [0] * len(detections_list)  #index=index of detection in list, value=believed corresponding true light
        possible_gt_list = copy.deepcopy(dataset.annotations_test[image_path])

        for ind, detection in enumerate(detections_list):
            closest = [[0,{}]]  # Initialize list keeps track of closest feature to detection - its distance + values in annotations_list form
            removal_index = 0  #Index of possible ground truth that will be removed after each iteration

            #Enumerate through each GT detect to find which is most likely the feature of interest
            for index, true_light in enumerate(possible_gt_list):
                feature_dist = get_image_distance([true_light["x_min"],true_light["y_min"],true_light["x_max"],true_light["y_max"]],
                                   [detection.bounding_box.left, detection.bounding_box.top,
                                   detection.bounding_box.left + detection.bounding_box.width,
                                   detection.bounding_box.top + detection.bounding_box.height])
                if feature_dist > closest[0] :
                    closest[0] = feature_dist
                    closest[1] = true_light
                    removal_index = index


            #Set the true BB associated with detection and, if appropiate, delete gorund truth possibilty
            truth_to_predict_map[ind] = closest[1]
            if len(detections_list) <= len(dataset.annotations_test[image_path]):
                del possible_gt_list[removal_index]

        #Compute statistics about how accruate predicitons are
        accuracy_stats = {}  #key=index in detecitons_list. value = dictionary of stat name to value. Stats include
        #BB overlap, whether class type is correct

        for index, detection in enumerate(detections_list):
            accuracy_stats[index] = {}
            accuracy_stats[index]["correct_class"] = (detection.predicted_classes[0] == truth_to_predict_map[index]["class"])
            accuracy_stats[index]["bounding_box_overlap"] = compute_bb_overlap([detection.bounding_box.left, detection.bounding_box.top,
                                   detection.bounding_box.left + detection.bounding_box.width, detection.bounding_box.top + detection.bounding_box.height],
                                                                              [truth_to_predict_map[index]["x_min"],truth_to_predict_map[index]["y_min"],
                                                                               truth_to_predict_map[index]["x_max"], truth_to_predict_map[index]["y_max"]])



        # Convert list of detections into same format as ground truth annotations
        processed_detections_list = []
        for detection in detections_list:
            processed_detections_list.append(
                {"class": detection.predicted_classes, "x_min": detection.bounding_box.left,
                 "y_min": detection.bounding_box.top,
                 "x_max": detection.bounding_box.left + detection.bounding_box.width,
                 "y_max": detection.bounding_box.top + detection.bounding_box.height})
        detections_list = processed_detections_list


        # Assign predictions to real detections




        # Compare accuracy of those detections to ground truth
        real_test_img_annotations = annotations_dict[image_path]
        real_class_name = annotations_dict[image_path][0]["class"]
        real_bounding_box = model.Bounds2D(annotations_dict[image_path][0]["x_min"],
                                           annotations_dict[image_path][0]["y_max"],
                                           annotations_dict[image_path][0]["x_max"] - annotations_dict[image_path][0][
                                               "x_min"],
                                           annotations_dict[image_path][0]["y_max"] - annotations_dict[image_path][0][
                                               "y_min"])



if __name__ == '__main__':
    benchmark_model("LISA")
'''

# Note: Currently, the classes retuned from the PredictedObject can only be a single class, cause that's all the classifier is designed for

heads, tail = os.path.split(image)
img_path = os.path.join(root_img_path, (str(i) + ".png"))
root_img_path = os.path.join(config.RESOURCES_ROOT, ("haar/data/" + dataset_name + "/images/"))  '''