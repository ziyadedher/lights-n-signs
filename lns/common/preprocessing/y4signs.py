import os
from collections import Counter


from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common import config
# from lns.common.preprocess import Preprocessor
from lns.common.dataset import Dataset 

DATASET_NAME = "Y4Signs"
PER_CLASS_LIMIT = 0  # PER_CLASS_LIMIT annotations per class, for testing
IMG_WIDTH = config.IMG_WIDTH
IMG_HEIGHT = config.IMG_HEIGHT


class Y4signs:
    def __init__(self, dataset_name, per_class_limit, img_width, img_height):
        self.dataset_name = dataset_name
        self.per_class_limit = per_class_limit
        self.img_width = img_width
        self.img_height = img_height

    def getDataset(self, path: str) -> Dataset:

        # use the same dataset for both but look for word train or test in either.
        is_train = True # default is train.
        is_full = False
        if "test" in path:
            path = path[:len(path) - len("_test")]
            is_train = False
        elif "_train" in path:
            if "sam" in path:
                path = path[:len(path) - len("_train_sam")]
            elif "manav" in path:
                path = path[:len(path) - len("_train_manav")]
            elif "helen" in path:
                path = path[:len(path) - len("_train_helen")]
            elif "matthieu" in path:
                path = path[:len(path) - len("_train_matthieu")]
            else:
                path = path[:len(path) - len("_train")]
        else:
            is_full = True


        if is_train:
            print("Path: " + path + " for training")
        else:
            print("Path: " + path + " for testing")

        
        if not os.path.isdir(path):
            raise FileNotFoundError("Could not find Y4Signs dataset on path: " + path)
        
        images: Dataset.Images = []
        classes: Dataset.Classes = []
        annotations: Dataset.Annotations = {}
        
        # list of classes [str] in set01/obj.names
        try:
            classes = open(os.path.join(path, "set01", "obj.names")).read().splitlines()
        except FileNotFoundError:
            print("Could not find obj.names in set01")

        list_of_sets = os.listdir(path)
        total_annotations = 0

        class_limit  = Counter()
        class_stats = Counter()
        
        
        for set_ in list_of_sets:
            f = open(os.path.join(path, set_, "train.txt"))

            #list of all absolute paths from train.txt. extra /data out
            for rel_image_path in f.read().split():
                # absolute image path to the image.
                abs_image_path = os.path.join(path, set_, rel_image_path[5:])
                # annotations file for corresponding image
                annotation_file_path = abs_image_path[:-4] + ".txt"

                try:
                    annotation_file = open(annotation_file_path)
                except FileNotFoundError:
                    print("annotation file " + annotation_file_path + " missing...skipping")
                    continue # continue inner loop

                temp_annotations = []

                classnums = Counter()
                flag = True
                # read the annotations file and loop through each line.
                for bounding_box in annotation_file.read().splitlines():
                    # split each line with spaces to get the [class, x1, y1, x2, y2]
                    annotation_arr = bounding_box.strip().split(" ")                
                    # check if legal annotation
                    if not len(annotation_arr) == 5:
                        raise InvalidBoundingBoxError(annotation_file_path)
                    

                    label_class = int(annotation_arr[0])
                    xcent = int(float(annotation_arr[1]) * self.img_width)
                    ycent = int(float(annotation_arr[2]) * self.img_height)
                    width = int(float(annotation_arr[3]) * self.img_width) 
                    height = int(float(annotation_arr[4]) * self.img_height)

                    xmin = int(xcent - 0.5*width)
                    ymin = int(ycent - 0.5*height)
                    # Construct and append an Object2D object
                    temp_annotations.append(Object2D(Bounds2D(
                        xmin,
                        ymin,
                        width,
                        height,
                    ), 
                    label_class))

                    class_limit[label_class] += 1
                    classnums[label_class] += 1

                    flag = flag and (class_limit[label_class] >= self.per_class_limit)

                # flag = False signifies that the image along with annotations should go to the testbatch.
                
                if (flag and is_train) or (not flag and not is_train) or is_full: 
                    for key in classnums:
                        class_stats[key]+=classnums[key]

                    total_annotations += len(temp_annotations)
                    images.append(abs_image_path)
                    
                    # add(abs_image_path, temp_annotation: list) key value pair.
                    annotations[abs_image_path] = temp_annotations


        print("Annotation stats: ")
        for key in class_stats:
            print(str(classes[key]) + ": " + str(class_stats[key]))

        
        if is_train:
            print("Dataset type: training")
        else:
            print("Dataset type: test")

        print("Found " + str(len(classes)) + " classes")
        print("Found " + str(len(images)) + " images")
        

        print("\nTotal annotations: " + str(total_annotations))

        
        return Dataset(self.dataset_name, images, classes, annotations)




class InvalidBoundingBoxError(Exception):
    """Exception raised for errors in the input path.

    Attributes:
        path -- input path which caused the error
    """

    def __init__(self, path):
        self.path = path
    
    def __str__(self):
        return "invalid bounding box at: " + self.path


preprocessor = Y4signs(DATASET_NAME, per_class_limit=PER_CLASS_LIMIT, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _Y4Signs_test(path: str) -> Dataset:
    return preprocessor.getDataset(path=path)





# import os
# from collections import Counter

# from lns.common.dataset import Dataset
# from lns.common.structs import Object2D, Bounds2D
# from lns.common.preprocess import Preprocessor



# DATASET_NAME = "Y4Signs_1036_584"
# IMG_WIDTH = 1036
# IMG_HEIGHT = 584

# @Preprocessor.register_dataset_preprocessor(DATASET_NAME)
# def _Y4Signs(path: str) -> Dataset:
#     print("Path: " + path)
    
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
#     print("Found " + str(len(list_of_sets)) + " sets")

#     classnums = Counter()
#     total_annotations = 0
    
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

#             # read the annotations file and loop through each line.
#             for bounding_box in annotation_file.read().splitlines():
#                 # split each line with spaces to get the [class, x1, y1, x2, y2]
#                 annotation_arr = bounding_box.strip().split(" ")                
#                 # check if legal annotation
#                 if not len(annotation_arr) == 5:
#                     raise InvalidBoundingBoxError(annotation_file_path)
                
#                 # Construct and append an Object2D object
#                 temp_annotations.append(Object2D(Bounds2D(
#                     int(float(annotation_arr[1]) * IMG_WIDTH),
#                     int(float(annotation_arr[2]) * IMG_HEIGHT),
#                     int(float(annotation_arr[3]) * IMG_WIDTH),
#                     int(float(annotation_arr[4]) * IMG_HEIGHT),
#                 ), 
#                 int(annotation_arr[0])))

#                 classnums[int(annotation_arr[0])] += 1
#                 total_annotations += 1
            
#             # append the image path to list of images
#             images.append(abs_image_path)
            
#             # exit for loop and added (abs_image_path, temp_annotation: list) key value pair.
#             annotations[abs_image_path] = temp_annotations

#     print("Found " + str(len(classes)) + " classes")
#     print("Found " + str(len(images)) + " images")

#     print("Annotations: ")
#     for i in range(len(classes)):
#         print(classes[i] + ": " + str(classnums[i]))
    
#     print("\nTotal annotations: " + str(total_annotations))

    
#     return Dataset(DATASET_NAME, images, classes, annotations)

#     # for i, class_name in enumerate(os.listdir(path)):
#     #     classes.append(class_name)
#     #     class_folder = os.path.join(path, class_name)
#     #     for file in os.listdir(class_folder):
#     #         image_path = os.path.join(class_folder, file)
#     #         images.append(image_path)
#     #         annotations[image_path] = [Object2D(Bounds2D(0, 0, 0, 0), i)]


# class InvalidBoundingBoxError(Exception):
#     """Exception raised for errors in the input path.

#     Attributes:
#         path -- input path which caused the error
#     """

#     def __init__(self, path):
#         self.path = path
    
#     def __str__(self):
#         return "invalid bounding box at: " + self.path

