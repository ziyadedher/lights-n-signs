"""Common preprocessing step for available datasets.

Manages the generation of a JSON file corresponding to each datasets that
contains an intermediate form of the data annotations for easier use within
the various subpackages representing each method of detection.

Each dataset's generated file's structure will be very similar to encourage
generalization of training methods to multiple datasets. A summary of the
structure of the generated file is below.

{
    "images": [
        "/absolute/path/to/image.png",
        "/absolute/path/to/other_image.png",
        "/absolute/path/to/nested/image.png",
        ...
    ],

    "classes": [
        "class_one",
        "other_detection_class",
        ...
    }

    "annotations": {
        "/absolute/path/to/image.png": [
            {
                "class": index_of_class,
                "x_min": x_coordinate_of_top_left_corner,
                "y_min": y_coordinate_of_top_left_corner,
                "x_max": x_coordinate_of_bottom_right_corner,
                "y_max": y_coordinate_of_bottom_right_corner
            },
            ...
        ],
        ...
    }
}
"""
from typing import Dict

from lns_common.config import Data
from lns_common.preprocess import preprocessing
from lns_common.preprocess.preprocessing import Dataset

from PIL import Image
import os


class Preprocessor:
    """Manager for all preprocessing and retrieval of preprocessed data."""

    # Preprocessor allocation for each dataset
    _PREPROCESSORS = {
        "LISA": preprocessing.preprocess_LISA,
        "Bosch": preprocessing.preprocess_bosch,
        "Custom": preprocessing.preprocess_custom,
        "Custom_testing": preprocessing.preprocess_custom,
        "sim": preprocessing.preprocess_sim,
        "mturk": preprocessing.preprocess_mturk,
        "cities": preprocessing.preprocess_cities,
        "KITTI": preprocessing.preprocess_KITTI
    }

    _preprocessing_data: Dict[str, Dataset] = {}

    @classmethod
    def preprocess(cls, dataset_name: str,
                   force: bool = False,
                   scale: float = 1.0, **kwargs) -> Dataset:
        """Preprocess the dataset with the given name and return the result.

        Setting <force> to `True` will force a preprocessing even if the
        preprocessed data already exists in memory.

        Raises `NoPreprocessorException` if a preprocessor for the dataset does
        not exist.
        """
        # Uses memoization to speed up preprocessing acquisition
        if not force and dataset_name in cls._preprocessing_data:
            return cls._preprocessing_data[dataset_name]

        dataset_path = Data.get_dataset_path(dataset_name)

        try:
            _preprocessor = cls._PREPROCESSORS[dataset_name]
        except KeyError:
            raise NoPreprocessorException(
                "Dataset {} has no allocated preprocessor."
            )

        preprocessed = _preprocessor(dataset_path, **kwargs)
        cls._preprocessing_data[dataset_name] = preprocessed

        if scale != 1.0:
            preprocessed = cls.fix_scale(dataset_path, preprocessed, scale)

        return preprocessed

    @classmethod
    def fix_scale(cls, dataset_path: str,
                  data: Dataset, scale: float) -> Dataset:
        """Downsizes the inputted dataset.
        """
        images: List[str] = data.images
        detection_classes: List[str] = data.classes
        annotations: Dict[str, List[Dict[str, int]]] = data.annotations

        new_dir = dataset_path + "_{}".format(scale)

        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

        for path, list_annotations in data.annotations.items():
            for index in range(len(list_annotations)):
                annotations[path][index]['x_min'] = \
                    int(annotations[path][index]['x_min'] * scale)
                annotations[path][index]['y_min'] = \
                    int(annotations[path][index]['y_min'] * scale)
                annotations[path][index]['x_max'] = \
                    int(annotations[path][index]['x_max'] * scale)
                annotations[path][index]['y_max'] = \
                    int(annotations[path][index]['y_max'] * scale)

            annos = annotations[path]
            del annotations[path]
            new_path = os.path.join(new_dir, os.path.basename(path))
            annotations[new_path] = annos

        for name, info in images.items():
            for image_index in range(len(info)):
                basename = os.path.basename(info[image_index])
                print(os.path.join(new_dir, basename))

                if os.path.isfile(os.path.join(new_dir, basename)):
                    continue

                img = Image.open(info[image_index])
                width, height = img.size
                width = int(scale * width)
                height = int(scale * height)
                img = img.resize((width, height), Image.ANTIALIAS)
                img.save(os.path.join(new_dir, basename))
                info[image_index] = os.path.join(new_dir, basename)

        return Dataset(data.name, images, detection_classes, annotations)


class NoPreprocessorException(Exception):
    """Raised when a dataset does not have an allocated preprocessor."""

    pass
