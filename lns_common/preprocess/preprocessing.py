"""Manages internal preprocessing methods.

Contains dedicated preprocessing functions for each dataset and the
preprocessing data structure.
"""
from typing import NewType, Dict, List, Tuple, Callable

import os
import csv
import copy
import yaml  # XXX: this could be sped up by using PyYaml C-bindings
import random
import pickle
import ast
import json
import urllib.request
import itertools
import statistics
from PIL import Image
from xml.etree import ElementTree as ET
import statistics

from lns_common import config


class Dataset:
    """Read-only container structure for data generated by preprocessing."""

    _name: str

    __images: Dict[str, List[str]]
    __classes: List[str]
    __annotations: Dict[str, List[Dict[str, int]]]

    def __init__(self, name: str,
                 images: Dict[str, List[str]], classes: List[str],
                 annotations: Dict[str, List[Dict[str, int]]]) -> None:
        """Initialize the data structure.

        <name> is a unique name for this dataset.
        <images> is a mapping of dataset name to list of absolute paths to the
        images in the dataset.
        <classes> is an indexed list of classes
        <annotations> is a mapping of image path to a list of "detections"
        represented by a dictionary containing keys `class` corresponding
        to the class index detected, `x_min`, `y_min` corresponding to the
        x-coordinate and y-coordinate of the top left corner of the bounding
        box, and `x_max`, `y_max` corresponding to the x-coordinate and
        y-coordinate of the bottom right corner of the bounding box.
        """
        self._name = name
        self.__images = images
        self.__classes = classes
        self.__annotations = annotations

    @property
    def name(self) -> str:
        """Get the name of this dataset."""
        return self._name

    @property
    def images(self) -> Dict[str, List[str]]:
        """Get a list of paths to all images available in the dataset."""
        return copy.deepcopy(self.__images)

    @property
    def classes(self) -> List[str]:
        """Get a mapping of ID to name for all classes in the dataset."""
        return copy.deepcopy(self.__classes)

    @property
    def annotations(self) -> Dict[str, List[Dict[str, int]]]:
        """Get all training image annotations.

        Image annotations are structured as a mapping of absolute image path
        (as given in `self.images`) to a list of detections. Each detection
        consists of a mapping from detection key to its respective information.

        Available detection keys are
        `class`, `x_min`, `y_min`, `x_max`, `y_max`.
        """
        return copy.deepcopy(self.__annotations)

    @property
    def test_annotations(self) -> Dict[str, List[Dict[str, int]]]:
        """Get all testing image annotations.

        Image annotations are structured as a mapping of absolute image path
        (as given in `self.images`) to a list of detections. Each detection
        consists of a mapping from detection key to its respective information.

        Available detection keys are
        `class`, `x_min`, `y_min`, `x_max`, `y_max`.
        """
        return copy.deepcopy(self.__annotations_test)

    def merge_classes(self, mapping: Dict[str, List[str]]) -> 'Dataset':
        """Get a new `Dataset` that has classes merged together.

        Merges the classes under the values in <mapping> under the class given
        by the respective key.
        """
        images = self.images
        classes = list(mapping.keys())
        annotations = self.annotations

        for path, annotation in annotations.items():
            for detection in annotation:
                # Change the detection class if required
                for new_class, mapping_classes in mapping.items():
                    if self.classes[detection["class"]] in mapping_classes:
                        detection["class"] = classes.index(new_class)
                        break

        return Dataset(self.name, images, classes, annotations)

    def __add__(self, other: 'Dataset') -> 'Dataset':
        """Magic method for adding two preprocessing data objects."""
        return Dataset(f"{self.name}-{other.name}",
                       {**self.images, **other.images},
                       self.classes + other.classes,
                       {**self.annotations, **other.annotations})


def preprocess_LISA(LISA_path: str, proportion: float = 1.0,
                    testset: bool = False) -> Dataset:
    """Preprocess and generate data for a LISA dataset at the given path.

    Only uses the `dayTrain` data subset.
    Raises `FileNotFoundError` if any of the required LISA files or folders
    is not found.
    """

    day_train_path = os.path.join(LISA_path, "dayTrain")
    if not os.path.isdir(day_train_path):
        raise FileNotFoundError("Could not find `dayTrain` in LISA dataset.")

    images: List[str] = []
    detection_classes: List[str] = []
    annotations: Dict[str, List[Dict[str, int]]] = {}

    for file_name in os.listdir(day_train_path):
        if not file_name.startswith("dayClip"):
            continue

        clip_path = os.path.join(day_train_path, file_name)
        frames_path = os.path.join(clip_path, "frames")
        annotations_path = os.path.join(clip_path, "frameAnnotationsBOX.csv")
        if not os.path.exists(frames_path):
            raise FileNotFoundError(
                f"Could not find frames folder {frames_path}."
            )
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(
                f"Could not find annotations file {annotations_path}"
            )

        # Register all the images
        for image_name in os.listdir(frames_path):
            images.append(os.path.join(frames_path, image_name))

        # Read annotations
        with open(annotations_path, "r") as annotations_file:
            reader = csv.reader(annotations_file, delimiter=";")
            for i, row in enumerate(reader):
                # Skip the first row, it is just headers
                if i == 0:
                    continue

                image_name = row[0].split("/")[-1]
                image_path = os.path.join(frames_path, image_name)
                detection_class = row[1]

                # Calculate the position and dimensions of the bounding box
                x_min = int(row[2])      # x-coordinate of top left corner
                y_min = int(row[3])      # y-coordinate of top left corner
                x_max = int(row[4])      # x-coordinate of bottom right corner
                y_max = int(row[5])      # y-coordinate of bottom right corner

                # Get the class index if it has already been registered
                # otherwise register it and select the index
                try:
                    class_index = detection_classes.index(detection_class)
                except ValueError:
                    class_index = len(detection_classes)
                    detection_classes.append(detection_class)

                # Package the detection
                if image_path not in annotations:
                    annotations[image_path] = []
                annotations[image_path].append({
                    "class": class_index,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                })
    annotations, images = delete_empties(annotations, images)

    if not testset:
        return set_proportions("LISA", {"LISA": images}, detection_classes,
                               annotations, proportion)
    else:
        return create_testset("cities", {"cities": images},
                              detection_classes, annotations, proportion)

def preprocess_lights(lights_path: str, proportion: float = 1.0,
                      testset: bool = False) -> Dataset:
    """Preprocess and generate data for our custom dataset at the given path.

    Raises `FileNotFoundError` if any of the required files or folders is not
    found.
    """
    images: List[str] = []
    detection_classes: List[str] = []
    annotations: Dict[str, List[Dict[str, int]]] = {}

    if not os.path.isdir(os.path.join(lights_path, "images_new")):
        os.mkdir(os.path.join(
            lights_path, "images_new"
        ))

    annotation_files: List[str] = os.listdir(lights_path)

    if len(annotation_files) == 0:
        raise FileNotFoundError(
            f"Could not find annotations file {directory}."
        )

    files_created: List[str] = os.listdir(os.path.join(
        lights_path, "images_new"
    ))

    for f in annotation_files:
        if os.path.isdir(os.path.join(lights_path, f)):
            continue

        print("Processing files for {}".format(f))

        with open(os.path.join(lights_path, f)) as csv_file:
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
                    url = "https://drive.google.com/uc?id={}&export=download".format(img_id)
                    print(url)
                    if "{}.png".format(img_id) not in files_created:
                        urllib.request.urlretrieve(
                            url, "{}.png".format(os.path.join(
                                lights_path, 'images_new', img_id
                            ))
                        )
                        print("{}.png Downloaded".format(img_id))

                        files_created.append("{}.png".format(img_id))

                    image_path = os.path.abspath(
                        os.path.join(
                            lights_path, "images_new", "{}.png".format(img_id)
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

                    images.append(image_path)

    annotations, images = delete_empties(annotations, images)

    annotations = annotation_fix(annotations)

    if not testset:
        return set_proportions("lights", {"lights": images},
                               detection_classes, annotations, proportion)
    else:
        return create_testset("lights", {"lights": images},
                              detection_classes, annotations, proportion)



def preprocess_sim(sim_path: str, proportion: float = 1.0,
                   testset: bool = False) -> Dataset:
    """Proprocess and generate data for a simulated dataset at the given path.

    Raises `FileNotFoundError` if any of the required sim files or folders is
    not found. Requires a `data.pkl` in the dataset generated by the script
    included with the dataset.
    """
    annotations_path = os.path.join(sim_path, 'data.pkl')
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(
            f"Could not find annotations file {annotations_path}."
        )

    DATA_TYPE = NewType(
        "DATA_TYPE", Dict[str, List[Tuple[str, Tuple[int, int, int, int]]]]
    )
    with open(annotations_path, "rb") as file:
        data: DATA_TYPE = pickle.load(file)

    images: List[str] = []
    detection_classes: List[str] = []
    annotations: Dict[str, List[Dict[str, int]]] = {}

    for image in data.keys():
        image_path = os.path.join(sim_path, image)
        print(image_path)
        images.append(image_path)
        for item in data[image]:
            item_class = item[0]
            bb = item[1]

            try:
                class_index = detection_classes.index(item_class)
            except ValueError:
                class_index = len(detection_classes)
                detection_classes.append(item_class)

            if image not in annotations:
                annotations[image_path] = []

            annotations[image_path].append({
                "class": class_index,
                "x_min": bb[0],
                "y_min": bb[1],
                "x_max": bb[0] + bb[2],
                "y_max": bb[1] + bb[3]
            })
    annotations, images = delete_empties(annotations, images)

    if not testset:
        return set_proportions("sim", {"sim": images}, detection_classes,
                               annotations, proportion)
    else:
        return create_testset("sim", {"sim": images},
                              detection_classes, annotations, proportion)


def preprocess_bosch(bosch_path: str, proportion: float = 1.0,
                     testset: bool = False) -> Dataset:
    """Preprocess and generate data for a Bosch dataset at the given path.

    Raises `FileNotFoundError` if any of the required Bosch files or
    folders is not found.
    """
    annotations_path = os.path.join(bosch_path, "train.yaml")
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(
            f"Could not find annotations file {annotations_path}."
        )
    with open(annotations_path, "r") as file:
        raw_annotations = yaml.load(file)

    images: List[str] = []
    detection_classes: List[str] = []
    annotations: Dict[str, List[Dict[str, int]]] = {}

    for annotation in raw_annotations:
        detections = annotation["boxes"]
        image_path = os.path.abspath(
            os.path.join(bosch_path, annotation["path"])
        )
        images.append(image_path)

        for detection in detections:
            label = detection["label"]
            x_min = round(detection["x_min"])
            x_max = round(detection["x_max"])
            y_min = round(detection["y_min"])
            y_max = round(detection["y_max"])

            # Get the class index if it has already been registered
            # otherwise register it and select the index
            try:
                class_index = detection_classes.index(label)
            except ValueError:
                class_index = len(detection_classes)
                detection_classes.append(label)

            # Package the detection
            if image_path not in annotations:
                annotations[image_path] = []
            annotations[image_path].append({
                "class": class_index,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            })
    annotations, images = delete_empties(annotations, images)

    if not testset:
        return set_proportions("Bosch", {"Bosch": images},
                               detection_classes, annotations, proportion)
    else:
        return create_testset("Bosch", {"Bosch": images},
                              detection_classes, annotations, proportion)


def preprocess_mturk(mturk_path: str, proportion: float = 1.0,
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
                        urllib.request.urlretrieve(
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

    annotations, images = delete_empties(annotations, images)
    if not testset:
        return set_proportions("mturk", {"mturk": images},
                               detection_classes, annotations, proportion)
    else:
        return create_testset("mturk", {"mturk": images},
                              detection_classes, annotations, proportion)


def preprocess_custom(custom_path: str, proportion: float = 1.0,
                      testset: bool = False) -> Dataset:
    """Preprocess and generate data for our custom dataset at the given path.

    Raises `FileNotFoundError` if any of the required files or folders is not
    found.
    """
    images: List[str] = []
    detection_classes: List[str] = []
    annotations: Dict[str, List[Dict[str, int]]] = {}

    # Go through all files in the directory
    for file_name in os.listdir(custom_path):
        # Skip any files that are not annotations
        if not file_name.endswith(".xml"):
            continue

        # Find the absolute image and label paths
        image_path = os.path.join(
            custom_path, file_name.split(".")[0] + ".jpg"
        )
        label_path = os.path.join(
            custom_path, file_name.split(".")[0] + ".xml"
        )

        # Make sure both the image and the label both exist
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Could not find annotations file {image_path}."
            )
        if not os.path.isfile(label_path):
            raise FileNotFoundError(
                f"Could not find annotations file {label_path}."
            )

        images.append(image_path)

        label_root = ET.parse(label_path).getroot()
        for label_object in label_root.findall("object"):
            # XXX: Probably clean up this code
            # Check that the data is in the correct form and that all nodes
            # exist correctly in the file
            bounding_box_node = label_object.find("bndbox")
            label_node = label_object.find("name")
            if bounding_box_node is None or label_node is None:
                continue
            xmin_node = bounding_box_node.find("xmin")
            xmax_node = bounding_box_node.find("xmax")
            ymin_node = bounding_box_node.find("ymin")
            ymax_node = bounding_box_node.find("ymax")
            if (
                xmin_node is None or xmax_node is None or
                ymin_node is None or ymax_node is None or
                xmin_node.text is None or xmax_node.text is None or
                ymin_node.text is None or ymax_node.text is None or
                label_node.text is None
            ):
                continue

            # Get all the values from the XML
            label = label_node.text
            x_min = round(float(xmin_node.text))
            x_max = round(float(xmax_node.text))
            y_min = round(float(ymin_node.text))
            y_max = round(float(ymax_node.text))

            # Get the class index if it has already been registered
            # otherwise register it and select the index
            try:
                class_index = detection_classes.index(label)
            except ValueError:
                class_index = len(detection_classes)
                detection_classes.append(label)

            # Package the detection
            if image_path not in annotations:
                annotations[image_path] = []
            annotations[image_path].append({
                "class": class_index,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            })

    annotations, images = delete_empties(annotations, images)

    if not testset:
        return set_proportions("Custom", {"Custom": images},
                               detection_classes, annotations, proportion)
    else:
        return create_testset("Custom", {"Custom": images},
                              detection_classes, annotations, proportion)


def preprocess_cities(cities_path: str, proportion: float = 1.0,
                      testset: bool = False) -> Dataset:
    images: List[str] = []
    detection_classes: List[str] = []
    annotations: Dict[str, List[Dict[str, int]]] = {}

    if not os.path.isdir(os.path.join(cities_path, "images_new")):
        os.mkdir(os.path.join(
            cities_path, "images_new"
        ))

    annotation_files: List[str] = os.listdir(cities_path)

    if len(annotation_files) == 0:
        raise FileNotFoundError(
            f"Could not find annotations file {directory}."
        )

    files_created: List[str] = os.listdir(os.path.join(
        cities_path, "images_new"
    ))

    for f in annotation_files:
        if os.path.isdir(os.path.join(cities_path, f)):
            continue

        print("Processing files for {}".format(f))

        with open(os.path.join(cities_path, f)) as csv_file:
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
                    url = "https://drive.google.com/uc?id={}&export=download".format(img_id)
                    print(url)
                    if "{}.png".format(img_id) not in files_created:
                        urllib.request.urlretrieve(
                            url, "{}.png".format(os.path.join(
                                cities_path, 'images_new', img_id
                            ))
                        )
                        print("{}.png Downloaded".format(img_id))

                        files_created.append("{}.png".format(img_id))

                    image_path = os.path.abspath(
                        os.path.join(
                            cities_path, "images_new", "{}.png".format(img_id)
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

                    images.append(image_path)

    annotations, images = delete_empties(annotations, images)

    annotations = annotation_fix(annotations)

    if not testset:
        return set_proportions("cities", {"cities": images},
                               detection_classes, annotations, proportion)
    else:
        return create_testset("cities", {"cities": images},
                              detection_classes, annotations, proportion)

def delete_empties(annotations: Dict[str, List[Dict[str, int]]], \
                   images: Dict[str, List[str]]) -> Tuple:
    delete_keys = []
    for key, anno in annotations.items():
        if len(anno) == 0:
            print("deleting {}".format(key))
            delete_keys.append(key)

    for d in delete_keys:
        del annotations[d]
        del images[images.index(d)]

    return annotations, images
    
def preprocess_KITTI(KITTI_path: str) -> Dataset:
    """Preprocess and generate data for our custom dataset at the given path.

    Raises `FileNotFoundError` if any of the required files or folders is not
    found.
    """

    images: List[str] = []
    detection_classes: List[str] = ['pedestrianCrossing', 'speedLimit15', 'speedLimit25', 'stop_', 'turnLeft', 'turnRight']
    annotations: Dict[str, List[Dict[str, int]]] = {}

    #Open up the csv file
    label_path = os.path.join(KITTI_path,
                              'relevantAnnotations.csv')  # Assume that CSV is with the rest of the directories
    labels_file = open(label_path, 'r')
    labels_reader = csv.DictReader(labels_file, fieldnames=['filename', 'annotations', 'bounding_box'])

    #iterate through the csv file to populate relevant structures
    for i, row in enumerate(labels_reader):
        if i == 0:   #First row is just header
            continue

        if row['annotations'] == 'stop':  #Error handling because of annotations processing
            class_name = 'stop_'
        else:
            class_name = row['annotations']

        img_name = row['filename'].split('/')[2]  #Add to the list of images
        file_name = os.path.join(os.path.join(KITTI_path, class_name), img_name)
        images.append(file_name)

        bounding_box = ast.literal_eval(row['bounding_box'])  #unpack the annotations
        x_min = int(bounding_box[0][0])
        y_min = int(bounding_box[0][1])
        x_max = int(bounding_box[1][0])
        y_max = int(bounding_box[1][1])
        temp = {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
        annotations[file_name] = [temp]

    return Dataset("KITTI_signs", {"KITTI_signs": images}, detection_classes, annotations)

def set_proportions(name: str,
                    images: Dict[str, List[str]], classes: List[str],
                    annotations: Dict[str, List[Dict[str, int]]],
                    proportion: float) -> Dataset:

    if proportion == 1.0:
        return Dataset(name, images, classes, annotations)

    new_annotations = {}

    print(images)

    for key, val in images.items():
        total = int(len(val) * proportion)

        random.seed(config.RAND_SEED)
        indices = random.sample([i for i in range(int(len(val)))], total)

        images[key] = [images[key][i] for i in indices]

        for path in images[key]:
            new_annotations[path] = annotations[path]

    return Dataset(name, images, classes, new_annotations)


def create_testset(name: str,
                   images: Dict[str, List[str]], classes: List[str],
                   annotations: Dict[str, List[Dict[str, int]]],
                   proportion: float) -> Dataset:

    if proportion == 1.0:
        return Dataset(name, images, classes, annotations)

    new_annotations = {}

    for key, val in images.items():
        total = int(len(val) * proportion)

        random.seed(config.RAND_SEED)
        inv_indices = sorted(random.sample(
            [i for i in range(int(len(val)))], total
        ))

        indices = []
        for i in range(int(len(val))):
            if len(inv_indices) != 0:
                if i == inv_indices[0]:
                    del inv_indices[0]
            else:
                indices.append(i)

        images[key] = [images[key][i] for i in indices]

        for path in images[key]:
            new_annotations[path] = annotations[path]

    return Dataset(name, images, classes, new_annotations)


def annotation_fix(annotations: Dict[str, List[Dict[str, int]]]
                   ) -> Dict:

    print("Fixing annotations...")

    new_annotations = {}

    for path, annotations in annotations.items():
        new_annotations[path] = []
        annos_to_process = annotations

        if len(annos_to_process) == 0:
            continue

        avg_x = sum([a["x_max"] - a["x_min"] for a in annos_to_process]) / len(annos_to_process)
        avg_y = sum([a["y_max"] - a["y_min"] for a in annos_to_process]) / len(annos_to_process)

        while len(annos_to_process) != 0:
            processing = []

            for i in range(len(annos_to_process)):
                if annos_to_process[i]["x_min"] > annos_to_process[i]["x_max"]:
                    annos_to_process[i]["x_min"],
                    annos_to_process[i]["x_max"] = \
                        annos_to_process[i]["x_max"], \
                        annos_to_process[i]["x_min"]

                if annos_to_process[i]["y_min"] > annos_to_process[i]["y_max"]:
                    annos_to_process[i]["y_min"],
                    annos_to_process[i]["y_max"] = \
                        annos_to_process[i]["y_max"], \
                        annos_to_process[i]["y_min"]

                if intersection_area(annos_to_process[i], annos_to_process[0]):
                    processing.append((annos_to_process[i], i))

            if len(processing) >= 2:
                ratio_func = lambda x: (x[0]['y_max'] - x[0]['y_min']) ** 0.5 / \
                    (x[0]['x_max'] - x[0]['x_min'])

                processing = sorted(processing, key=ratio_func)

                box = processing[-1]

                new_annotations[path].append(box[0])

                processing = sorted(processing, key=lambda x: -x[1])

            for i in range(len(processing)):
                del annos_to_process[processing[i][1]]

    return new_annotations


def intersection_area(annotation1: Dict[str, int],
                      annotation2: Dict[str, int]) -> int:

        xmin = max(annotation1["x_min"], annotation2["x_min"])
        ymin = max(annotation1["y_min"], annotation2["y_min"])
        xmax = min(annotation1["x_max"], annotation2["x_max"])
        ymax = min(annotation1["y_max"], annotation2["y_max"])

        if annotation1["x_min"] > annotation2["x_max"]:
            return False

        if annotation2["x_min"] > annotation1["x_max"]:
            return False

        if annotation1["y_min"] > annotation2["y_max"]:
            return False

        if annotation2["y_min"] > annotation1["y_max"]:
            return False

        return True