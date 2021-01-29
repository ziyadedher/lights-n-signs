import os

from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor


DATASET_NAME = "Y4Signs"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _Y4Signs(path: str) -> Dataset:
    

    if not os.path.isdir(path):
        raise FileNotFoundError("Could not find Y4Signs dataset on this path.")
    
    images: Dataset.Images = []
    classes: Dataset.Classes = []
    annotations: Dataset.Annotations = {}
    
    # list of classes [str] in set01/obj.names
    classes = open(os.path.join(path, "set01", "obj.names")).read().splitlines()


    
    for set_ in os.listdir(path):
        if os.path.isdir(os.path.join(path, set_, "train.txt")):
            raise FileNotFoundError("train.txt does not exist in " + set_)
        
        f = open(os.path.join(path, set_, "train.txt"))
        #list of all absolute paths from train.txt. extra /data out
        for rel_image_path in f.read().split():
            # absolute image path to the image.
            abs_image_path = os.path.join(path, set_, rel_image_path[4:])

            # append the image path to list of images
            images.append(abs_image_path)

            # annotations file for corresponding image
            annotation_file_path = abs_image_path[:-4] + ".txt"
            annotation_file = open(annotation_file_path)

            temp_annotations = []

            # read the annotations file and loop through each line.
            for bounding_box in annotation_file.read().splitlines():
                # split each line with spaces to get the [class, x1, y1, x2, y2]
                annotation_arr = bounding_box.strip().split(" ")
                
                # check if legal
                if not len(annotation_arr) == 5:
                    raise InvalidBoundingBoxError(annotation_file_path)
                
                # Construct and append an Object2D object
                temp_annotations.append(Object2D(Bounds2D(
                    float(annotation_arr[1]),
                    float(annotation_arr[2]),
                    float(annotation_arr[3]),
                    float(annotation_arr[4]),
                ), 
                int(annotation_arr[0])))
            
            # exit for loop and added (abs_image_path, temp_annotation: list) key value pair.
            annotations[abs_image_path] = temp_annotations
    
    return Dataset(DATASET_NAME, images, classes, annotations)

    # for i, class_name in enumerate(os.listdir(path)):
    #     classes.append(class_name)
    #     class_folder = os.path.join(path, class_name)
    #     for file in os.listdir(class_folder):
    #         image_path = os.path.join(class_folder, file)
    #         images.append(image_path)
    #         annotations[image_path] = [Object2D(Bounds2D(0, 0, 0, 0), i)]


class InvalidBoundingBoxError(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(self, path):
        self.path = path
    
    def __str__(self):
        return "invalid bounding box at: " + self.path

