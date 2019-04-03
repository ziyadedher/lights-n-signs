from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor

import pickle
import os

DATASET_NAME = "scale_signs"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def preprocess_scale_lights(scale_signs_path: str, proportion: float = 1.0,
                            testset: bool = False) -> Dataset:
    """Preprocess and generate data for our custom dataset at the given path.

    Raises `FileNotFoundError` if any of the required files or folders is not
    found.
    """
    with open(os.path.join(scale_signs_path, 'scale_signs.pickle'), "rb") as f:
        data = pickle.load(f)

    images: List[str] = data['images']
    detection_classes: List[str] = data['classes']
    annotations: Dict[str, List[Dict[str, int]]] = data['annotations']

    d = Dataset(DATASET_NAME, {DATASET_NAME: images}, detection_classes, annotations)
    #print(data)
    return d

