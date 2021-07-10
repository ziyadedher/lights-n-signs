import os
import random
import cv2
import pickle
from collections import Counter
from tqdm import tqdm  # type: ignore
from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor
from lns.common import config


# things to configure-------------------
DATASET_SRC = 'Y4Signs_filtered_1036_584_train_split' # can change the source of data.
IMG_WIDTH = config.IMG_WIDTH # to change IMG_WIDTH to resize images, please change in config
IMG_HEIGHT = config.IMG_HEIGHT # to change IMG_HEIGHT to resize images, please change in config
NEW_SIZE = (IMG_WIDTH, IMG_HEIGHT)
DO_NOT_SPLIT = False # True if you don't want to split but only augment and/or resize
PER_CLASS_LIMIT = config.PER_CLASS_LIMIT # does not matter if DO_NOT_SPLIT = True. To change PER_CLASS_LIMIT please change in config

# can name dest datasets anything.
if DO_NOT_SPLIT:
    DATASET_DEST_TRAIN = '/home/od/.lns-training/resources/data/Y4Signs_filtered_{0}_{1}_whole'.format(IMG_WIDTH, IMG_HEIGHT)
else:
    DATASET_DEST_TRAIN = '/home/od/.lns-training/resources/data/Y4Signs_filtered_{0}_{1}_train_split'.format(IMG_WIDTH, IMG_HEIGHT)
    DATASET_DEST_TEST = '/home/od/.lns-training/resources/data/Y4Signs_filtered_{0}_{1}_test_split'.format(IMG_WIDTH, IMG_HEIGHT)
#-----------------




random.seed(10)

print('preprocessing raw dataset...')
Preprocessor.preprocess(DATASET_SRC, force=True)
# preprocessor = Y4signs(dataset_name = "Temp", per_class_limit=0, img_width = IMG_WIDTH, img_height = IMG_HEIGHT) 

if (not os.path.exists(DATASET_DEST_TRAIN)) and (not os.path.exists(DATASET_DEST_TEST)):
    os.mkdir(DATASET_DEST_TRAIN)
    if not DO_NOT_SPLIT:
        os.mkdir(DATASET_DEST_TEST)
else:
    print("Destination folders already exist. Please delete those or rename destination folders.")
    exit(1)


raw_dataset: Dataset = preprocessor.getDataset(DATASET_SRC)

random.shuffle(raw_dataset.images) # shuffle order of images.


class_limit = Counter()
class_stats_train = Counter()


class_stats_test = None
if not DO_NOT_SPLIT
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

        annots[0].class_index
        annots[0].bounds.width
        
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
            new_annot_path = os.path.join(DATASET_DEST_TRAIN, "{}.pkl".format(train_img_counter))
            train_img_counter += 1
        else:
            new_img_path = os.path.join(DATASET_DEST_TEST, "{}.png".format(test_img_counter))
            new_annot_path = os.path.join(DATASET_DEST_TEST, "{}.pkl".format(test_img_counter))
            test_img_counter += 1
        

        resized = cv2.resize(raw_img, NEW_SIZE, interpolation = cv2.INTER_AREA)

        # augmentation code
        # annots = annotations for image in question
        # annots.bounds.left/top/width/height for coordinates.
        # resized = numpy array for image 


        cv2.imwrite(new_img_path, resized) # resized image goes to new dataset path

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
