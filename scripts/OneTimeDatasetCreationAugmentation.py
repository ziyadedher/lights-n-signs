import os
import random
import cv2
import pickle
import math 
import numpy as np
from collections import Counter
from tqdm import tqdm  # type: ignore
from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor
from lns.common import config
from lns.common.preprocessing.y4signs_2x import Y4Signs_2x
from lns.common.structs import Bounds2D # For recreate 2D Bound
# from lns.common.preprocessing.y4signs import Y4signs

# things to configure-------------------
DATASET_SRC = 'Y4Signs_filtered_1036_584_train_split' # can change the source of data.
IMG_WIDTH = config.IMG_WIDTH # to change IMG_WIDTH to resize images, please change in config
IMG_HEIGHT = config.IMG_HEIGHT # to change IMG_HEIGHT to resize images, please change in config
NEW_SIZE = (IMG_WIDTH, IMG_HEIGHT)
DO_NOT_SPLIT = True # True if you don't want to split but only augment and/or resize
PER_CLASS_LIMIT = config.PER_CLASS_LIMIT # does not matter if DO_NOT_SPLIT = True. To change PER_CLASS_LIMIT please change in config
OUTPUT_PATH = '/home/lns/aaron_w/dataset/Y4Signs_filtered_{0}_{1}_whole'.format(IMG_WIDTH, IMG_HEIGHT)
#Data Augmentation parameters
MAX_ROTATION_ANGLE_DEGREE = 10
MAX_BRIGHTNESS_CHANGE = 70

#---------Data Augmentation Functions--------
# Randomly change the brightness of imgs
def increase_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # generate random brightness noise
    value = random.randint(-MAX_BRIGHTNESS_CHANGE, MAX_BRIGHTNESS_CHANGE+1)

    # Clap the result to 0 - 255
    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        value = int(-value)
        lim = 0 + value
        v[v < lim] = 0
        v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

# Randomly resize the images. Note that this will not crop the annotataions of the images
def resize_image(img, annots):
    if len(annots) == 0:
        # If no annotations, then dont do anything
        return img, annots

    # Find the max and min positions of the annotations. Make sure we dont crop the annotations out
    annot_max_x = annot_max_y = 0
    annot_min_x = annot_min_y = float('inf')
    for annot in annots:
        x1, y1 = annot.bounds.left, annot.bounds.top
        x2, y2 = annot.bounds.right, annot.bounds.bottom
        annot_max_x = max(annot_max_x, x2)
        annot_max_y = max(annot_max_y, y2)
        annot_min_x = min(annot_min_x, x1)
        annot_min_y = min(annot_min_y, y1)

    # generate a random cropping window. Make sure don't crop the annotations
    annot_min_x = max(0, annot_min_x)
    annot_min_y = max(0, annot_min_y)
    annot_max_x = min(annot_max_x, IMG_WIDTH-1)
    annot_max_y = min(annot_max_y, IMG_HEIGHT-1)
    
    new_x1 = random.randint(0, annot_min_x)
    new_x2 = random.randint(annot_max_x, IMG_WIDTH)
    new_y1 = random.randint(0, annot_min_y)
    new_y2 = random.randint(annot_max_y, IMG_HEIGHT)

    img = img[new_y1: new_y2, new_x1: new_x2]

    # Change annotation positions. Change bounds by creating a new bound
    for annot in annots:
        annot.bounds = Bounds2D(annot.bounds.left - new_x1, annot.bounds.top - new_y1, annot.bounds.width, annot.bounds.height)
    
    return img, annots

# Randomly rotate the images. The bounding boxes might be rotated out of the img.
def rotate_image(image, annots):

    def rotate_point(point, center, angle_rad):
        x, y = point
        x0, y0 = center
        x1 = x0+(x-x0)*math.cos(angle_rad)+(y-y0)*math.sin(angle_rad)
        y1 = y0-(x-x0)*math.sin(angle_rad)+(y-y0)*math.cos(angle_rad)
        return (x1, y1)

    def maxOf4(a,b,c,d):
        return max(max(a,b),max(c,d))

    def minOf4(a,b,c,d):
        return min(min(a,b),min(c,d))

    if len(annots) == 0:
        # If no annotations, then dont do anything
        return image, annots

    # Rotate the image
    angle = random.randint(-MAX_ROTATION_ANGLE_DEGREE, MAX_ROTATION_ANGLE_DEGREE+1)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    # Rotate the annotation
    x0, y0 = image_center
    angle_rad = math.radians(angle)
    for annot in annots:
        x1, y1 = annot.bounds.left, annot.bounds.top
        x2, y2 = annot.bounds.right, annot.bounds.bottom

        # Rotate each corner of the bounding box
        left_top_x, left_top_y = rotate_point((x1, y1), (x0, y0), angle_rad)
        left_bot_x, left_bot_y = rotate_point((x1, y2), (x0, y0), angle_rad)
        right_top_x, right_top_y = rotate_point((x2, y1), (x0, y0), angle_rad)
        right_bot_x, right_bot_y = rotate_point((x2, y2), (x0, y0), angle_rad)

        # Create new bounding box
        annot_max_x = maxOf4(left_top_x, left_bot_x, right_top_x, right_bot_x)
        annot_max_y = maxOf4(left_top_y, left_bot_y, right_top_y, right_bot_y)
        annot_min_x = minOf4(left_top_x, left_bot_x, right_top_x, right_bot_x)
        annot_min_y = minOf4(left_top_y, left_bot_y, right_top_y, right_bot_y)

        width = annot_max_x - annot_min_x
        height = annot_max_y - annot_min_y

        annot.bounds = Bounds2D(math.ceil(annot_min_x), math.ceil(annot_min_y), math.ceil(width), math.ceil(height))

    return rotated_img, annots


# Add annotations onto imgages
def visualize_annot(imgs, annots):
    # For testing purpose: visualize image with annotations
    for annot in annots:
        x1, y1 = annot.bounds.left, annot.bounds.top
        x2, y2 = annot.bounds.right, annot.bounds.bottom
        imgs = cv2.rectangle(imgs,(x1,y1),(x2,y2),(0,255,0),2)
    return imgs


# can name dest datasets anything.
if DO_NOT_SPLIT:
    DATASET_DEST_TRAIN = '/home/od/.lns-training/resources/data/Y4Signs_filtered_{0}_{1}_whole'.format(IMG_WIDTH, IMG_HEIGHT)
else:
    DATASET_DEST_TRAIN = '/home/od/.lns-training/resources/data/Y4Signs_filtered_{0}_{1}_train_split'.format(IMG_WIDTH, IMG_HEIGHT)
    DATASET_DEST_TEST = '/home/od/.lns-training/resources/data/Y4Signs_filtered_{0}_{1}_test_split'.format(IMG_WIDTH, IMG_HEIGHT)

DATASET_DEST_TRAIN = OUTPUT_PATH
DATASET_DEST_TEST = DATASET_DEST_TRAIN
#-----------------




random.seed(10)

print('preprocessing raw dataset...')
Preprocessor.preprocess(DATASET_SRC, force=True)
# preprocessor = Y4signs(dataset_name = DATASET_SRC, per_class_limit=0, img_width = IMG_WIDTH, img_height = IMG_HEIGHT) 
preprocessor = Y4Signs_2x(dataset_name = DATASET_SRC) 
os.makedirs(DATASET_DEST_TRAIN, exist_ok=True)
# if (not os.path.exists(DATASET_DEST_TRAIN)) and (not os.path.exists(DATASET_DEST_TEST)):
#     # os.mkdir(DATASET_DEST_TRAIN)
#     os.makedirs(DATASET_DEST_TRAIN, exist_ok=True)
#     if not DO_NOT_SPLIT:
#         # os.mkdir(DATASET_DEST_TEST)
#         os.makedirs(DATASET_DEST_TEST, exist_ok=True)
# else:
#     print("Destination folders already exist. Please delete those or rename destination folders.")
#     exit(1)


raw_dataset: Dataset = preprocessor.getDataset('/home/od/.lns-training/resources/data/Y4Signs_filtered_1036_584_train_split')

# random.shuffle(raw_dataset.images) # shuffle order of images.


class_limit = Counter()
class_stats_train = Counter()


class_stats_test = None
if not DO_NOT_SPLIT:
    class_stats_test = Counter()

aspect_ratio_sums = {} # to get the average aspect ratio per class.
for cl in range(len(raw_dataset.classes)):
    aspect_ratio_sums[cl] = 0.0


train_img_counter = 0
if not DO_NOT_SPLIT:
    test_img_counter = 0

with tqdm(desc="Processing", total=len(raw_dataset.images), miniters=1) as tqdm_bar:
    for raw_img_path in raw_dataset.images: # in shuffled order
        # update bar
        tqdm_bar.update()

        # read img from raw dataset
        raw_img = cv2.imread(raw_img_path, cv2.IMREAD_UNCHANGED)
        # read annots from raw dataset. These annots will not change 
        # in the new dataset as they were created with the given image size
        annots = raw_dataset.annotations[raw_img_path]

        # annots[0].class_index
        # annots[0].bounds.width
        
        is_train = True
        for annot in annots: # check if this image should go in test or train dataset
            label_class = annot.class_index 
            class_limit[label_class] += 1
            is_train = is_train and (class_limit[label_class] >= PER_CLASS_LIMIT)
        
        for annot in annots: # for stats.
            label_class = annot.class_index 
            aspect_ratio = float(annot.bounds.width) / float(annot.bounds.width)

            if is_train:
                aspect_ratio_sums[label_class] += aspect_ratio
                class_stats_train[label_class] += 1
            else:
                if not DO_NOT_SPLIT: 
                    class_stats_test[label_class] += 1


        # to name image path
        if is_train or DO_NOT_SPLIT: # if DO_NOT_SPLIT is True, all data will be in the train folder.
            new_img_path = os.path.join(DATASET_DEST_TRAIN, "{}.png".format(train_img_counter))
            ori_img_path = os.path.join(DATASET_DEST_TRAIN, "{}_origin.png".format(train_img_counter))
            new_annot_path = os.path.join(DATASET_DEST_TRAIN, "{}.pkl".format(train_img_counter))
            train_img_counter += 1
        else:
            new_img_path = os.path.join(DATASET_DEST_TEST, "{}.png".format(test_img_counter))
            new_annot_path = os.path.join(DATASET_DEST_TEST, "{}.pkl".format(test_img_counter))
            test_img_counter += 1
        

        resized = cv2.resize(raw_img, NEW_SIZE, interpolation = cv2.INTER_AREA)

        # -------------------augmentation ----------------------
        # annots = annotations for image in question
        # annots.bounds.left/top/width/height for coordinates.
        # resized = numpy array for image

        # Change Brightness
        resized = increase_brightness(resized)

        # Rotate
        resized, annots = rotate_image(resized, annots)

        # Crop Images
        resized, annots = resize_image(resized, annots)

        # Visualize Annotations
        resized = visualize_annot(resized, annots)

        cv2.imwrite(new_img_path, resized) # resized image goes to new dataset path
        cv2.imwrite(ori_img_path, raw_img)

        with open(new_annot_path, 'wb') as annot_file:
            pickle.dump(annots, annot_file) # dump annotation as pkl file.


with open(os.path.join(DATASET_DEST_TRAIN, 'classes.pkl'), 'wb') as classes_file:
    pickle.dump(raw_dataset.classes, classes_file) # dump classes list as pkl file.

if not DO_NOT_SPLIT:
    with open(os.path.join(DATASET_DEST_TEST, 'classes.pkl'), 'wb') as classes_file:
        pickle.dump(raw_dataset.classes, classes_file) # dump classes list as pkl file.

train_stats = open(os.path.join(DATASET_DEST_TRAIN, "stats.txt"), "w") 
if not DO_NOT_SPLIT:
    test_stats = open(os.path.join(DATASET_DEST_TEST, "stats.txt"), "w") 

# train stats
train_stats.write("Annotation stats: \n")
for key in class_stats_train:
    train_stats.write(str(raw_dataset.classes[key]) + ": " + str(class_stats_train[key]) + "\n")

train_stats.write("Annotation avg aspect ratios: \n")
for key in class_stats_train:
    avg_aspect_ratio = float(aspect_ratio_sums[key] / float(class_stats_train[key]))
    train_stats.write(str(raw_dataset.classes[key]) + ": " + str(avg_aspect_ratio) + "\n")

if not DO_NOT_SPLIT:
    # annotation stats for test set
    test_stats.write("Annotation stats: \n")
    for key in class_stats_test:
        test_stats.write(str(raw_dataset.classes[key]) + ": " + str(class_stats_test[key]) + "\n")


