from typing import List

import os
import csv
import urllib

from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor


DATASET_NAME = "cities"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _cities(path: str) -> Dataset:
    """Preprocess and generate data for the cities dataset at the given path.

    Requires tjat the folder be in the resources area and that the csv annotations
      already be there
    Raises `FileNotFoundError` if any of the required LISA files or folders
    is not found.
    """
    images: Dataset.Images = {DATASET_NAME: []}
    classes: Dataset.Classes = []
    annotations: Dataset.Annotations = {}

    if not os.path.isdir(os.path.join(path, "images_new")):
        os.mkdir(os.path.join(
            path, "images_new"
        ))

    annotation_files: List[str] = os.listdir(path)
    if not annotation_files:
        raise FileNotFoundError(f"Could not find annotations file {directory}.")

    files_created: List[str] = os.listdir(os.path.join(
        path, "images_new"
    ))

    for f in annotation_files:
        if os.path.isdir(os.path.join(path, f)):
            continue

        print("Processing files for {}".format(f))

        with open(os.path.join(path, f)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            id_index = 0
            annos = 0

            for row in csv_reader:
                if id_index == 0:
                    for r in range(len(row)):
                        if "Answer.annotatedResult.boundingBoxes" in row[r]:
                            annos = r
                        if "Input.image_id" in row[r]:
                            id_index = r
                else:
                    img_id = str(row[id_index])
                    url = "https://drive.google.com/uc?id={}&export=download".format(img_)
                    print(url)
                    if "{}.png".format(img_id) not in files_created:
                        urllib.request.urlretrieve(
                            url, "{}.png".format(os.path.join(
                                path, 'images_new', img_id
                            ))
                        )
                        print("{}.png Downloaded".format(img_id))

                        files_created.append("{}.png".format(img_id))

                    image_path = os.path.abspath(
                        os.path.join(
                            path, "images_new", "{}.png".format(img_id)
                        )
                    )

                    data = json.loads(row[annos])

                    if image_path not in annotations:
                        annotations[image_path] = []

                    for d in data:
                        c = d["label"]

                        if c not in detection_classes:
                            detection_classes.append(c)

                        x = d["left"]
                        y = d["top"]
                        w = d["width"]
                        h = d["height"]

                        if config.MIN_SIZE > w * h:
                            continue

                        annotations[image_path].append({
                            "class": detection_classes.index(c),
                            "x_min": x,
                            "y_min": y,
                            "x_max": x + w,
                            "y_max": y + h
                        })

                    images[DATASET_NAME].append(image_path)

    annotations = annotation_fix(annotations)

    return Dataset(DATASET_NAME, images, classes, annotations)
