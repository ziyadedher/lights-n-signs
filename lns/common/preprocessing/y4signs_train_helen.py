from lns.common.preprocessing.y4signs import Y4signs
from lns.common.preprocess import Preprocessor
from lns.common.dataset import Dataset
from lns.common import config



DATASET_NAME = "Y4Signs_1036_584_train_helen"
PER_CLASS_LIMIT = config.PER_CLASS_LIMIT
IMG_WIDTH = config.IMG_WIDTH
IMG_HEIGHT = config.IMG_HEIGHT

preprocessor = Y4signs(DATASET_NAME, per_class_limit=PER_CLASS_LIMIT, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _Y4Signs_train(path: str) -> Dataset:
    return preprocessor.getDataset(path=path)
    








# import os
# from collections import Counter

# from lns.common.dataset import Dataset
# from lns.common.structs import Object2D, Bounds2D
# from lns.common.preprocess import Preprocessor

# DATASET_NAME = "Y4Signs_1036_584_train"
# PER_CLASS_LIMIT = 150  # PER_CLASS_LIMIT annotations per class, for testing
# IMG_WIDTH = 1036
# IMG_HEIGHT = 584

# @Preprocessor.register_dataset_preprocessor(DATASET_NAME)
# def _Y4Signs(path: str) -> Dataset:

#     # use the same dataset for both but look for word train or test in either.
#     is_train = True # default is train.
#     if("test" in path):
#         path = path[:len(path) - len("_test")]
#         is_train = False
#     else:
#         path = path[:len(path) - len("_train")]

#     if is_train:
#         print("Path: " + path + " for training")
#     else:
#         print("Path: " + path + " for testing")

    
#     if not os.path.isdir(path):
#         raise FileNotFoundError("Could not find Y4Signs dataset on this path.")
    
#     images: Dataset.Images = []
#     classes: Dataset.Classes = []
#     annotations: Dataset.Annotations = {}
    
#     # list of classes [str] in set01/obj.names
#     try:
#         classes = open(os.path.join(path, "set01", "obj.names")).read().splitlines()
#     except FileNotFoundError:
#         print("Could not find obj.names in set01")

#     list_of_sets = os.listdir(path)
#     total_annotations = 0

#     class_limit  = Counter()
#     class_stats = Counter()
    
    
#     for set_ in list_of_sets:
#         f = open(os.path.join(path, set_, "train.txt"))

#         #list of all absolute paths from train.txt. extra /data out
#         for rel_image_path in f.read().split():
#             # absolute image path to the image.
#             abs_image_path = os.path.join(path, set_, rel_image_path[5:])
#             # annotations file for corresponding image
#             annotation_file_path = abs_image_path[:-4] + ".txt"
#             try:
#                 annotation_file = open(annotation_file_path)
#             except FileNotFoundError:
#                 print("annotation file " + annotation_file_path + " missing")
#                 continue

#             temp_annotations = []

#             classnums = Counter()
#             flag = True
#             # read the annotations file and loop through each line.
#             for bounding_box in annotation_file.read().splitlines():
#                 # split each line with spaces to get the [class, x1, y1, x2, y2]
#                 annotation_arr = bounding_box.strip().split(" ")                
#                 # check if legal annotation
#                 if not len(annotation_arr) == 5:
#                     raise InvalidBoundingBoxError(annotation_file_path)
                

#                 label_class = int(annotation_arr[0])
#                 # Construct and append an Object2D object
#                 temp_annotations.append(Object2D(Bounds2D(
#                     int(float(annotation_arr[1]) * IMG_WIDTH),
#                     int(float(annotation_arr[2]) * IMG_HEIGHT),
#                     int(float(annotation_arr[3]) * IMG_WIDTH),
#                     int(float(annotation_arr[4]) * IMG_HEIGHT),
#                 ), 
#                 label_class))

#                 class_limit[label_class] += 1
#                 classnums[label_class] += 1

#                 flag = flag and (class_limit[label_class] >= PER_CLASS_LIMIT)

#             # flag = False signifies that the image along with annotations should go to the testbatch.
            
#             if (flag and is_train) or (not flag and not is_train): 
#                 for key in classnums:
#                     class_stats[key]+=classnums[key]

#                 total_annotations += len(temp_annotations)
#                 images.append(abs_image_path)
                
#                 # add(abs_image_path, temp_annotation: list) key value pair.
#                 annotations[abs_image_path] = temp_annotations


#     print("Annotation stats: ")
#     for key in class_stats:
#         print(str(classes[key]) + ": " + str(class_stats[key]))

    
#     if is_train:
#         print("Dataset type: training")
#     else:
#         print("Dataset type: test")

#     print("Found " + str(len(classes)) + " classes")
#     print("Found " + str(len(images)) + " images")
    

#     print("\nTotal annotations: " + str(total_annotations))

    
#     return Dataset(DATASET_NAME, images, classes, annotations)




# class InvalidBoundingBoxError(Exception):
#     """Exception raised for errors in the input path.

#     Attributes:
#         path -- input path which caused the error
#     """

#     def __init__(self, path):
#         self.path = path
    
#     def __str__(self):
#         return "invalid bounding box at: " + self.path




