"""Compute object distances from camera across a dataset.

Contains functions to compute object distances, and remove outliers
(i.e. objects too far from camera)
"""

import os
import struct

from typing import Dict, List, Tuple

from lns.common import config
from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor
from lns.common.resources import Resources

class DatasetDistance:
    """Tracks distances of objects from camera within a dataset."""

    _datasets: Dict[str, Dataset]
    _dataset_distances: Dict[str, Dict[str, List[Tuple[int, float]]]]

    def __init__(cls):
        Preprocessor.init_cached_preprocessed_data()
        cls._datasets = Preprocessor._preprocessed_data


    def compute_cache(cls):
        cls.compute_distances()
        cls.cache_distances()

    def cache_distances(cls):
        """Cache object distances on disk."""
        for name in cls._dataset_distances:
            distances_pkl = os.path.join(Resources.get_root(), config.DATA_DISTANCES_FOLDER_NAME, name + config.PKL_EXTENSION)
            distances = cls._dataset_distances[name]
            with open(distances_pkl, "wb") as file:
                pickle.dump(distances, file)


    def compute_distances(cls):
        """Compute distances."""
        for dataset_name in cls._datasets:
            print(dataset_name)
            cls.distance(dataset_name)


    def distance(cls, dataset_name: str) -> None:
        """Return a mapping of images to a list of object distances."""
        distances: Dict[str, List[Tuple[int, float]]] = {}

        dataset = cls._datasets[dataset_name]

        for image in dataset.images:
            print(image)
            distances[image] = []
            img_area = _img_area(image)

            for detection in dataset.annotations[image]:
                distances[image].append((detection.class_index, detection.bounds.area / img_area))

        cls._dataset_distances[dataset_name] = distances


    def _img_area(cls, img_name: str) -> float:
        """Determine dimensions of image stored at absolute path <img_name>."""
        width, height = _get_image_size(img_name)
        return width * height


    # https://stackoverflow.com/questions/15800704/get-image-size-without-loading-image-into-memory
    # https://raw.githubusercontent.com/scardine/image_size/master/get_image_size.py
    class UnknownImageFormat(Exception):
        """Exception for parsing metadata of image in unknown format."""


    def _get_image_size(cls, file_path):
        """Return (width, height) for a given img file content w/ no external dependencies."""
        size = os.path.getsize(file_path)

        with open(file_path, "rb") as input_file:
            height = -1
            width = -1
            data = input_file.read(25)

            if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
                # GIFs
                raw_w, raw_h = struct.unpack("<HH", data[6:10])
                width = int(raw_w)
                height = int(raw_h)
            elif ((size >= 24) and data.startswith(b'\211PNG\r\n\032\n')
                  and (data[12:16] == 'IHDR')):
                # PNGs
                raw_w, raw_h = struct.unpack(">LL", data[16:24])
                width = int(raw_w)
                height = int(raw_h)
            elif (size >= 16) and data.startswith(b'\211PNG\r\n\032\n'):
                # older PNGs?
                raw_w, raw_h = struct.unpack(">LL", data[8:16])
                width = int(raw_w)
                height = int(raw_h)
            elif (size >= 2) and data.startswith(b'\377\330'):
                # JPEG
                width, height = _process_jpeg(input_file)
            else:
                raise UnknownImageFormat(
                    "Sorry, don't know how to get information from this file."
                )

        return width, height


    def _process_jpeg(cls, input_file):
        """Process metadata of JPEG."""
        msg = " raised while trying to decode as JPEG."
        input_file.seek(0)
        input_file.read(2)
        binary = input_file.read(1)
        try:
            while binary and ord(binary) != 0xDA:
                while ord(binary) != 0xFF:
                    binary = input_file.read(1)
                while ord(binary) == 0xFF:
                    binary = input_file.read(1)
                if (ord(binary) >= 0xC0 and ord(binary) <= 0xC3):
                    input_file.read(3)
                    raw_h, raw_w = struct.unpack(">HH", input_file.read(4))
                    break
                else:
                    input_file.read(int(struct.unpack(">H", input_file.read(2))[0]) - 2)
                binary = input_file.read(1)
            width = int(raw_w)
            height = int(raw_h)
        except struct.error:
            raise UnknownImageFormat("StructError" + msg)
        except ValueError:
            raise UnknownImageFormat("ValueError" + msg)
        except Exception as ex:
            raise UnknownImageFormat(ex.__class__.__name__ + msg)

        return width, height

