import os
import random
import cv2
import pickle
from collections import Counter
from tqdm import tqdm  # type: ignore
from lns.common.dataset import Dataset
from lns.common.preprocessing.y4signs import Y4signs

DATASET_DEST_TRAIN = '/home/od/.lns-training/resources/data/Y4Signs_2072_1168_train'
DATASET_DEST_TEST = '/home/od/.lns-training/resources/data/Y4Signs_2072_1168_test'
DATASET_SRC = '/home/od/.lns-training/resources/data/Y4Signs'
IMG_WIDTH = 2072
IMG_HEIGHT = 1168
NEW_SIZE = (IMG_WIDTH, IMG_HEIGHT)
PER_CLASS_LIMIT = 100

random.seed(10)

print('preprocessing raw dataset...')
preprocessor = Y4signs(dataset_name = "Temp", per_class_limit=0, img_width = IMG_WIDTH, img_height = IMG_HEIGHT)
# preprocesses with our new dimensions! Cheeky. 

if (not os.path.exists(DATASET_DEST_TRAIN)) and (not os.path.exists(DATASET_DEST_TEST)):
    os.mkdir(DATASET_DEST_TRAIN)
    os.mkdir(DATASET_DEST_TEST)
else:
    print("LMAO. There's already something at DESTINATION fam. Try another location!")
    exit(1)

raw_dataset: Dataset = preprocessor.getDataset(DATASET_SRC)

random.shuffle(raw_dataset.images) # shuffle order of images.
class_limit = Counter()
class_stats_train = Counter()
class_stats_test = Counter()

aspect_ratio_sums = {}
for cl in range(len(raw_dataset.classes)):
    aspect_ratio_sums[cl] = 0.0


train_img_counter = 0
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
                class_stats_test[label_class] += 1


        if is_train:
            new_img_path = os.path.join(DATASET_DEST_TRAIN, "{}.png".format(train_img_counter))
            new_annot_path = os.path.join(DATASET_DEST_TRAIN, "{}.pkl".format(train_img_counter))
            train_img_counter += 1
        else:
            new_img_path = os.path.join(DATASET_DEST_TEST, "{}.png".format(test_img_counter))
            new_annot_path = os.path.join(DATASET_DEST_TEST, "{}.pkl".format(test_img_counter))
            test_img_counter += 1
        

        resized = cv2.resize(raw_img, NEW_SIZE, interpolation = cv2.INTER_AREA)
        cv2.imwrite(new_img_path, resized) # resized image goes to new dataset path

        with open(new_annot_path, 'wb') as annot_file:
            pickle.dump(annots, annot_file) # dump annotation as pkl file.

with open(os.path.join(DATASET_DEST_TRAIN, 'classes.pkl'), 'wb') as classes_file:
    pickle.dump(raw_dataset.classes, classes_file) # dump annotation as pkl file.

with open(os.path.join(DATASET_DEST_TEST, 'classes.pkl'), 'wb') as classes_file:
    pickle.dump(raw_dataset.classes, classes_file) # dump annotation as pkl file.

train_stats = open(os.path.join(DATASET_DEST_TRAIN, "stats.txt"), "w") 
test_stats = open(os.path.join(DATASET_DEST_TEST, "stats.txt"), "w") 
train_stats.write("Annotation stats: \n")
for key in class_stats_train:
    train_stats.write(str(raw_dataset.classes[key]) + ": " + str(class_stats_train[key] + "\n"))

train_stats.write("Annotation avg aspect ratios: \n")
for key in class_stats_train:
    avg_aspect_ratio = float(aspect_ratio_sums[key] / float(class_stats_train[key]))
    train_stats.write(str(raw_dataset.classes[key]) + ": " + str(avg_aspect_ratio + "\n"))

# annotation stats for test set
test_stats.write("Annotation stats: \n")
for key in class_stats_test:
    test_stats.write(str(raw_dataset.classes[key]) + ": " + str(class_stats_test[key] + "\n"))
