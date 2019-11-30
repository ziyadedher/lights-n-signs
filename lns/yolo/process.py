"""Data processing for YOLOv3.

Manages all data processing for the generation of data ready to be trained on with our YOLOv3 training backend.
"""
import os
from typing import Iterable, Tuple

from PIL import Image  # type: ignore

from lns.common.dataset import Dataset
from lns.common.process import ProcessedData, Processor
from tqdm import tqdm  # type: ignore


class YoloData(ProcessedData):
    """Data container for all YOLOv3 processed data.

    Stores paths to all files needed by backend YOLOv3 training.
    """

    __classes: str
    __train_annotations: str
    __test_annotations: str

    def __init__(self, classes: str, train_annotations: str, test_annotations: str) -> None:
        """Initialize the data structure."""
        self.__classes = classes
        self.__train_annotations = train_annotations
        self.__test_annotations = test_annotations

    def get_classes(self) -> str:
        """Get the path to the class names file."""
        return self.__classes

    def get_train_annotations(self) -> str:
        """Get the path to the training annotations file."""
        return self.__train_annotations

    def get_test_annotations(self) -> str:
        """Get the path to the testing annotations file."""
        return self.__test_annotations


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
        train_annotations_path = os.path.join(processed_data_folder, "train_annotations")
        test_annotations_path = os.path.join(processed_data_folder, "test_annotations")
        with open(train_annotations_path, "w") as train_file, open(test_annotations_path, "w") as test_file:
            def order_label(label) -> Tuple[str, str, str, str, str]:
                return (str(label.class_index),
                        str(label.bounds.left), str(label.bounds.top),
                        str(label.bounds.right), str(label.bounds.bottom))

            def generate_annotations() -> Iterable[str]:
                for i, image in tqdm(enumerate(images), desc=f"YOLO Processing `{dataset.name}`"):
                    labels_str = " ".join(" ".join(order_label(label)) for label in annotations[image])
                    if not labels_str:
                        continue

                    width, height = 0, 0
                    with Image.open(image) as img:
                        width, height = img.size
                    yield f"{i} {image} {width} {height} {labels_str}\n"

            annotations_lines = list(generate_annotations())
            random.shuffle(annotations_lines)

            train_split = int(0.9 * len(annotations_lines))

            train_file.writelines(annotations_lines[:train_split])
            test_file.writelines(annotations_lines[:test_split])

        return YoloData(classes_path, train_annotations_path, test_annotations_path)


YoloProcessor.init_cached_processed_data()
