import os

from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor
from lns.common.preprocessing.y4signs_2x import Y4Signs_2x


DATASET_NAME = "Y4Signs_filtered_1036_584_train_split"

@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _Y4Signs_test_split(path: str) -> Dataset:
    preprocessor = Y4Signs_2x(DATASET_NAME)
    return preprocessor.getDataset(path)


