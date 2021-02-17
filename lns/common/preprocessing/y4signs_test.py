from lns.common.preprocessing.y4signs import Y4signs
from lns.common.preprocess import Preprocessor
from lns.common.dataset import Dataset



DATASET_NAME = "Y4Signs_1036_584_test"
PER_CLASS_LIMIT = 150  # PER_CLASS_LIMIT annotations per class, for testing
IMG_WIDTH = 1036
IMG_HEIGHT = 584

preprocessor = Y4signs(DATASET_NAME, per_class_limit=PER_CLASS_LIMIT, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _Y4Signs_test(path: str) -> Dataset:
    return preprocessor.getDataset(path=path)





