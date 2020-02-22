"""Arbitrary dataset representation.

Contains classes for the tracking of dataset objects through our pipeline
as well as simple utility functions for operation on the datasets.
"""

import collections
import copy
from typing import Dict, List, Tuple

import numpy as np  # type: ignore
from lns.common.structs import Object2D
from lns.common.utils.img_area import img_area
from lns.common.utils.kmeans import kmeans, avg_iou

_Images = List[str]
_Classes = List[str]
_Labels = List[Object2D]
_Annotations = Dict[str, _Labels]


class Dataset:
    """Dataset container structure for data generated by preprocessing."""

    Images = List[str]
    Classes = List[str]
    Labels = List[Object2D]
    Annotations = Dict[str, Labels]

    _name: str
    _dynamic: bool

    __images: Images
    __classes: Classes
    __annotations: Annotations

    def __init__(self, name: str, images: _Images, classes: _Classes, annotations: _Annotations, *,
                 dynamic=False) -> None:
        """Initialize the data structure.

        <name> is a unique name for this dataset.
        <images> is a list of absolute paths to the images in the dataset.
        <classes> is an indexed list of classes.
        <annotations> is a mapping of image path to a list of 2D objects present in the image.
        """
        self._name = name
        self._dynamic = dynamic
        self.__images = copy.deepcopy(images)
        self.__classes = copy.deepcopy(classes)
        self.__annotations = copy.deepcopy(annotations)

    @property
    def name(self) -> str:
        """Get the name of this dataset."""
        return self._name

    @property
    def dynamic(self) -> bool:
        """Return whether or not this dataset was dynamically generated."""
        return self._dynamic

    @property
    def images(self) -> _Images:
        """Get a list of paths to all images available in the dataset."""
        return self.__images

    @property
    def classes(self) -> _Classes:
        """Get a mapping of ID to name for all classes in the dataset."""
        return self.__classes

    @property
    def annotations(self) -> _Annotations:
        """Get all training image annotations.

        Image annotations are structured as a mapping of absolute image path
        (as given in `self.images`) to a list of Object2D.
        """
        return self.__annotations

    def merge_classes(self, mapping: Dict[str, List[str]]) -> 'Dataset':
        """Get a new `Dataset` that has classes merged together.

        Merges the classes under the values in <mapping> under the class given
        by the respective key.
        """
        images = self.images
        original_classes = self.classes
        classes = list(set(self.classes) - set(c for l in list(mapping.values()) for c in l) | set(mapping.keys()))
        annotations = self.annotations

        for image in images:
            for detection in annotations[image]:
                # Check if the detection class is in the new classes
                if original_classes[detection.class_index] in classes:
                    detection.class_index = classes.index(original_classes[detection.class_index])
                    continue

                # Change the detection class if required
                for new_class, mapping_classes in mapping.items():
                    if self.classes[detection.class_index] in mapping_classes:
                        detection.class_index = classes.index(new_class)
                        break

        return Dataset(self.name, images, classes, annotations, dynamic=True)

    def generate_anchors(self, num_clusters: int) -> List[List[float]]:
        """Return anchors (N, 2)."""

        def get_kmeans(boxes, num_clusters):
            anchors = kmeans(boxes, num_clusters)
            ave_iou = avg_iou(boxes, anchors)
            anchors = anchors.astype('int').tolist()
            anchors = sorted(anchors, key=lambda x: x[0] * x[1])
            return anchors, ave_iou

        def get_boxes_from_annotation(annotations):
            """Return a list of box (r, 2)."""
            boxes = []
            for file in annotations:
                for object2d in annotations[file]:
                    box = [object2d.bounds.width, object2d.bounds.height]
                    if box[0] <= 0 or box[1] <= 0 or np.isnan(box[0]) or np.isnan(box[1]):
                        print('invalid bounding box:{}'.format(box))
                        continue
                    boxes.append(np.array(box))
            return np.array(boxes)

        annotations = self.__annotations
        boxes = get_boxes_from_annotation(annotations)
        anchors, _ = get_kmeans(boxes, num_clusters)
        anchors = np.reshape(np.asarray(anchors, np.float32), [-1, 2])
        print("Anchors generated: {}".format(anchors))
        return anchors

    def prune(self, threshold: float, delete_empty=True) -> 'Dataset':
        """Return a new dataset with relative annotation sizes under a given <threshold> pruned."""
        dists: Dict[float, List[Tuple[str, Object2D]]] = collections.defaultdict(list)

        images = self.images
        classes = self.classes
        annotations = self.annotations

        for image in images:
            image_area = img_area(image)

            for detection in annotations[image]:
                dists[detection.bounds.area / image_area].append((image, detection))

        for dist in sorted(dists.keys()):
            if dist > threshold:
                break
            for image, detection in dists[dist]:
                if detection in annotations[image]:
                    annotations[image].remove(detection)

        # Delete images with no labels
        if delete_empty:
            remove_img_list = []
            for image in annotations:
                if not annotations[image]:
                    remove_img_list.append(image)
            for image in set(remove_img_list):
                if image in images:
                    images.remove(image)
                if image in annotations:
                    del annotations[image]

        return Dataset(self.name, images, classes, annotations, dynamic=True)

    def split(self, *props: List[float]) -> List['Dataset']:
        """Shuffles and partitions dataset into portions <props>."""
        props = np.array(props)
        if np.sum(props) != 1 or np.any(props <= 0):  # type: ignore
            raise ValueError("<props> must be strictly positive and sum to 1.")

        images = self.images

        inds = np.arange(len(images))
        np.random.shuffle(inds)

        splits: List[Dataset] = []

        ranges = np.insert(np.ceil(np.cumsum(props) * len(images)).astype(int), 0, 0)
        ranges[-1] = len(images) + 1

        for low, high in zip(ranges[:-1], ranges[1:]):
            new_images = list(np.array(images)[inds[low:high]])
            splits.append(
                Dataset(
                    "{}_{}_{}".format(self.name, low, high),
                    new_images,
                    self.classes,
                    {i: self.annotations[i] for i in new_images},
                    dynamic=True
                ))  # type: ignore

        return splits

    def __add__(self, other: 'Dataset') -> 'Dataset':
        """Magic method for adding two preprocessing data objects."""
        self_class_names = self.classes[:]
        other_class_names = other.classes[:]

        name = f"{self.name}-{other.name}"
        images = list(set(self.images + other.images))
        classes = list(set(self.classes + other.classes))

        self_annotations = copy.deepcopy(self.annotations)
        for annotation in self_annotations.values():
            for label in annotation:
                label.class_index = classes.index(self_class_names[label.class_index])
        other_annotations = copy.deepcopy(other.annotations)
        for annotation in other_annotations.values():
            for label in annotation:
                label.class_index = classes.index(other_class_names[label.class_index])

        for image, labels in other_annotations.items():
            if image in self_annotations:
                self_annotations[image].extend(labels)
            else:
                self_annotations[image] = labels
        dynamic = self.dynamic or other.dynamic

        return Dataset(name, images, classes, self_annotations, dynamic=dynamic)

    def __len__(self) -> int:
        """Magic method to get the length of this `Dataset`.

        We define the length of a dataset to the the total number of images.
        """
        return len(self.__images)

    def __eq__(self, other: object) -> bool:
        """Magic method to check if two datasets are equal."""
        if not isinstance(other, Dataset):
            raise NotImplementedError

        return (self.__images == other.images
                and self.__classes == other.classes
                and self.__annotations == other.annotations)

    def __copy__(self) -> 'Dataset':
        """Magic method to copy over this dataset."""
        return Dataset(self.name, self.images, self.classes, self.annotations, dynamic=True)
    __deepcopy__ = __copy__
