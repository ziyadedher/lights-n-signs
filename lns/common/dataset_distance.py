"""Compute object distances from camera across a dataset.

Contains functions to compute object distances, and remove outliers
(i.e. objects too far from camera)
"""

from typing import List, Tuple

from lns.common.dataset import Dataset

from PIL import Image

def distance(dataset: Dataset) -> Dict[str, List[Tuple[int, float]]]:
    """Return a mapping of images to a list of object distances.
    """
    to_return = {}

    for image in dataset.images:
        to_return[image] = []
        for detection in dataset.annotations[image]:
            to_return[image].append((detection.class_index,
                                     detection.bounds.area / _img_area(image)))

     return to_return

def _img_area(img_name: str) -> float:
    """Determine dimensions of image stored at absolute path <img_name>.
    """
    # TODO: compute this without loading in the whole image?
    im = Image.open(img_name)
    width, height = im.size
    return width * height
