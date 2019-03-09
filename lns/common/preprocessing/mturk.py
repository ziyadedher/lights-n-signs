from typing import List, Dict

import os
import csv
import ast
from urllib import request

from lns.common import config
from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor

DATASET_NAME = "mturk"

@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _mturk(mturk_path: str, proportion: float = 1.0,
                     testset: bool = False) -> Dataset:
    """Preprocess and generate data for our custom dataset at the given path.

    Raises `FileNotFoundError` if any of the required files or folders is not
    found.
    """
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

            for row in csv_reader:
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

    return Dataset(DATASET_NAME, {DATASET_NAME: images}, detection_classes, annotations)
