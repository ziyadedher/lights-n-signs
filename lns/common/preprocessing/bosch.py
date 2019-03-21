import os
import yaml  # XXX: this could be sped up by using PyYaml C-bindings

import numpy as np
import cv2 as cv
from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor
from lns.common.preprocessing.augment import augment

DATASET_NAME = "Bosch"
PRODUCT_DIM = 50
BACKGROUND_DIR = '/home/lns/lns/vinit/bg/'


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _bosch(path: str) -> Dataset:
    """Preprocess and generate data for a Bosch dataset at the given path.

    Raises `FileNotFoundError` if any of the required Bosch files or
    folders is not found.
    """
    
    backgrounds = [ 
        os.path.join(BACKGROUND_DIR, bg_name) 
        for bg_name in os.listdir(BACKGROUND_DIR)
    ]

    images: Dataset.Images = {DATASET_NAME: []}
    classes: Dataset.Classes = []
    annotations: Dataset.Annotations = {}

    annotations_path = os.path.join(path, "train.yaml")
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(f"Could not find annotations file {annotations_path}.")
    with open(annotations_path, "r") as file:
        raw_annotations = yaml.load(file)

    for annotation in raw_annotations:
        detections = annotation["boxes"]
        image_path = os.path.abspath(os.path.join(path, annotation["path"]))
        print(image_path)
        for detection in detections:
            label = detection["label"]
            x_min = round(detection["x_min"])
            x_max = round(detection["x_max"])
            y_min = round(detection["y_min"])
            y_max = round(detection["y_max"])

            # Get the class index if it has already been registered
            # otherwise register it and select the index
            try:
                class_index = classes.index(label)
            except ValueError:
                class_index = len(classes)
                classes.append(label)

            # Package the detection
            if image_path not in annotations:
                annotations[image_path] = []
                images[DATASET_NAME].append(image_path)
            annotations[image_path].append({
                "class": class_index,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            })

    augmented = []
    for image in images[DATASET_NAME][:10]:
        train_image = cv.imread(image)
        i = 1
        for bg in backgrounds[:PRODUCT_DIM]:
            for ant in annotations[image]:
                # Setup new path
                new_path = image + f".aug{i}.png"
                print(new_path)
                # Extract ROI
                x1, y1, x2, y2 = ant['x_min'], ant['y_min'], ant['x_max'], ant['y_max']
                sign_image = train_image[y1:y2, x1:x2]
                # Augment and modify class
                a = augment(sign_image, bg, new_path)
                if a is None: continue

                a['class'] = ant['class']
                # Add annotation
                annotations[new_path] = [a]
                # Save image path
                augmented.append(new_path)
                i += 1
        np.random.shuffle(backgrounds)
    
    images[DATASET_NAME].extend(augmented)
    return Dataset(DATASET_NAME, images, classes, annotations)
