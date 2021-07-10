"""Data processing for YOLOv3.

Manages all data processing for the generation of data ready to be trained on with our YOLOv3 training backend.
"""
import os
from typing import Iterable, Tuple

from PIL import Image  # type: ignore
from tqdm import tqdm  # type: ignore

from lns.common.dataset import Dataset
from lns.common.process import ProcessedData, Processor


class YoloData(ProcessedData):
    """Data container for all YOLOv3 processed data.

    Stores paths to all files needed by backend YOLOv3 training.
    """

    __classes: str
    __annotations: str

    def __init__(self, classes: str, annotations: str) -> None:
        """Initialize the data structure."""
        self.__classes = classes
        self.__annotations = annotations

    def get_classes(self) -> str:
        """Get the path to the class names file."""
        return self.__classes

    def get_annotations(self) -> str:
        """Get the path to the annotations file."""
        return self.__annotations


class YoloProcessor(Processor[YoloData]):
    """YOLOv3 processor responsible for data processing to YOLOv3-valid formats."""

    @classmethod
    def method(cls) -> str:
        """Get the training method this processor is for."""
        return "yolo"

    @classmethod
    def _process(cls, dataset: Dataset) -> YoloData:
        processed_data_folder = os.path.join(cls.get_processed_data_path(), dataset.name)

        classes = dataset.classes
        classes_path = os.path.join(processed_data_folder, "classes")
        with open(classes_path, "w") as classes_file:
            def generate_classes() -> Iterable[str]:
                for class_name in classes:
                    yield f"{class_name}\n"

            classes_file.writelines(generate_classes())

        images = dataset.images
        annotations = dataset.annotations
        annotations_path = os.path.join(processed_data_folder, "annotations")
        with open(annotations_path, "w") as annotations_file:
            def order_label(label) -> Tuple[str, str, str, str, str]:
                return (str(label.class_index),
                        str(label.bounds.left), str(label.bounds.top),
                        str(label.bounds.right), str(label.bounds.bottom))

            def generate_annotations() -> Iterable[str]:
                for i, image in tqdm(list(enumerate(images)), desc=f"YOLO Processing `{dataset.name}`"):
                    labels_str = " ".join(" ".join(order_label(label)) for label in annotations[image])
                    if not labels_str:
                        continue

                    width, height = 0, 0
                    with Image.open(image) as img:
                        width, height = img.size
                    yield f"{i} {image} {width} {height} {labels_str}\n"

            annotations_file.writelines(generate_annotations())

        return YoloData(classes_path, annotations_path)


YoloProcessor.init_cached_processed_data()
