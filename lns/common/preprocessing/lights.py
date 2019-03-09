from typing import List

import os
import csv
import json
from urllib import request

from lns.common import config
from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor
from lns.common.preprocessing.fix_dataset import annotation_fix


DATASET_NAME = "lights"


# XXX: does not work!
@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _lights(path: str) -> Dataset:
    """Preprocess and generate data for our custom dataset at the given path.

    Raises `FileNotFoundError` if any of the required files or folders is not
    found.
    """
    images: Dataset.Images = {DATASET_NAME: []}
    classes: Dataset.Classes = []
    annotations: Dataset.Annotations = {}

    if not os.path.isdir(os.path.join(path, "images_new")):
        os.mkdir(os.path.join(path, "images_new"))

    annotation_files: List[str] = os.listdir(path)
    files_created: List[str] = os.listdir(os.path.join(path, "images_new"))

    if len(annotation_files) == 0:
        raise FileNotFoundError(f"Could not find annotations file in {path}.")

    for file_path in annotation_files:
        if os.path.isdir(os.path.join(path, file_path)):
            continue

        with open(os.path.join(path, file_path)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            id_index = 0
            annos = 0

            for row in csv_reader:
                if id_index == 0:
                    for i, entry in enumerate(row):
                        if "Answer.annotatedResult.boundingBoxes" in entry:
                            annos = i
                        if "Input.image_id" in entry:
                            id_index = i
                else:
                    img_id = str(row[id_index])
                    url = "https://drive.google.com/uc?id={}&export=download".format(img_id)
                    if "{}.png".format(img_id) not in files_created:
                        request.urlretrieve(url, os.path.join(path, "images_new", f"{img_id}.png"))
                        files_created.append("{}.png".format(img_id))

                    image_path = os.path.abspath(os.path.join(path, "images_new", f"{img_id}.png"))

                    data = json.loads(row[annos])

                    if image_path not in annotations:
                        annotations[image_path] = []

                    for datum in data:
                        class_name = datum["label"]
                        if class_name not in classes:
                            classes.append(class_name)

                        x = datum["left"]
                        y = datum["top"]
                        w = datum["width"]
                        h = datum["height"]

                        if config.MIN_SIZE > w * h:
                            continue

                        annotations[image_path].append({
                            "class": classes.index(class_name),
                            "x_min": x,
                            "y_min": y,
                            "x_max": x + w,
                            "y_max": y + h
                        })

                    if image_path in annotations:
                        images[DATASET_NAME].append(image_path)

    return Dataset(DATASET_NAME, images, classes, annotation_fix(annotations))
