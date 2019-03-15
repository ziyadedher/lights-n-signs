from typing import List, Dict

import numpy as np
import cv2 as cv

import os
import csv
import ast
from urllib import request

from lns.common import config
from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor
from lns.common.preprocessing.augment import augment

DATASET_NAME = "mturk"
PRODUCT_DIM = 50
BACKGROUND_DIR = '/home/lns/lns/vinit/bg/'

@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _mturk(mturk_path: str, proportion: float = 1.0,
                     testset: bool = False) -> Dataset:
    """Preprocess and generate data for our custom dataset at the given path.

    Raises `FileNotFoundError` if any of the required files or folders is not
    found.
    """
    
    backgrounds = [ 
        os.path.join(BACKGROUND_DIR, bg_name) 
        for bg_name in os.listdir(BACKGROUND_DIR)
    ]

    images: List[str] = []
    detection_classes: List[str] = []
    annotations: Dict[str, List[Dict[str, int]]] = {}

    if not os.path.isdir(os.path.join(mturk_path, "images_new")):
        os.mkdir(os.path.join(
            mturk_path, "images_new"
        ))

    annotation_files: List[str] = os.listdir(mturk_path)

    if len(annotation_files) == 0:
        raise FileNotFoundError(
            f"Could not find annotations file {directory}."
        )

    files_created: List[str] = os.listdir(os.path.join(
        mturk_path, "images_new"
    ))

    for f in annotation_files:
        if os.path.isdir(os.path.join(mturk_path, f)):
            continue

        print("Processing files for {}".format(f))

        with open(os.path.join(mturk_path, f)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            url_index = 0
            id_index = -1
            annos = 0
            classification = 0
            orientation = 0
            count = 0
            for row in csv_reader:
                count += 1
                if count > 25: break
                if url_index == 0:
                    for r in range(len(row)):
                        if "Input.image_url" in row[r]:
                            url_index = r
                        if "Input.objects_to_find" in row[r]:
                            classification = r
                        if "Answer.annotation_data" in row[r]:
                            annos = r
                        if "Input.orientation" in row[r]:
                            orientation = r
                else:
                    img_id = str(row[id_index])
                    url = row[url_index]
                    print(url)
                    if "{}.png".format(img_id) not in files_created:
                        request.urlretrieve(
                            url, "{}.png".format(os.path.join(
                                mturk_path, 'images_new', img_id
                            ))
                        )
                        print("{}.png".format(img_id))

                    image_path = os.path.abspath(
                        os.path.join(
                            mturk_path, "images_new", "{}.png".format(img_id)
                        ))

                    if open(image_path, 'rb').read()[-2:] != b'\xff\xd9':
                        print("Error in image, skipping")
                        continue

                    c = row[classification]
                    if c not in detection_classes:
                        detection_classes.append(c)

                    labels = ast.literal_eval(row[annos])
                    to_orient = int(row[orientation]) == 6

                    valid = False

                    for label in labels:
                        x = label['left']
                        y = label['top']
                        w = label['width']
                        h = label['height']

                        if w * h < config.MIN_SIZE:
                            continue

                        if not valid:
                            valid = True
                            annotations[image_path] = []

                        annotations[image_path].append({
                            "class": detection_classes.index(c),
                            "x_min": x,
                            "y_min": y,
                            "x_max": x + w,
                            "y_max": y + h
                        })

                    if not valid:
                        continue

                    images.append(image_path)

    augmented = []
    for image in images:
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
    
    images.extend(augmented)

    return Dataset(DATASET_NAME, {DATASET_NAME: images}, detection_classes, annotations)
