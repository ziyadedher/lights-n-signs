import os

from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor


DATASET_NAME = "Y4Signs"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _Y4Signs(path: str) -> Dataset:
    images: Dataset.Images = []
    classes: Dataset.Classes = []
    annotations: Dataset.Annotations = {}
    num_images = 0
    for set_ in os.listdir(path):
        f = open(os.path.join(path, set_, "train.txt"))
        images.extend([for i in f.read().split()] i[4:]) #list of all images from train.txt. extra /data out
        
        classes.append()

    # for i, class_name in enumerate(os.listdir(path)):
    #     classes.append(class_name)
    #     class_folder = os.path.join(path, class_name)
    #     for file in os.listdir(class_folder):
    #         image_path = os.path.join(class_folder, file)
    #         images.append(image_path)
    #         annotations[image_path] = [Object2D(Bounds2D(0, 0, 0, 0), i)]

    return Dataset(DATASET_NAME, images, classes, annotations)