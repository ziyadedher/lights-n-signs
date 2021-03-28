from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor
from typing import List
import cv2 as cv # type: ignore
import numpy as np
import os
from tqdm import tqdm # type: ignore
from pathlib import Path

class SVMProcessor:
    def __init__(self, path: str, dataset: Dataset, compare: List[tuple]):
        """Handles preprocessing of dataset

        Args:
            path (str): [path to store preprocessed dataset]
            dataset (str): [path to dataset]
            compare (List[tuple]): [List of tuples containing class indices to compare]
            eg. [(1, (2, 3)), (2, (1, 3)), (3, (1, 2))]
        """
        self.dataset = dataset 
        self.path = path
        self.compare = compare
        self.splits = {}
        for a, b in compare:
            self.splits[a] = []
            for c in b: 
                self.splits[c] = []


    def preprocess(self, force: bool = True):
        if force or not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        else:
            print("Dataset already processed")
            return

        print("Creating crops...")
        with tqdm(desc="Processing", total=len(self.dataset.annotations.keys()), miniters=1) as tqdm_bar:
            # need_print = 1
            for image_path, labels in self.dataset.annotations.items():
                tqdm_bar.update()
                
                img_sz = Path(image_path).stat().st_size  # image size in bytes
                # ./speed_limit_20/IMG_20181007_152311-1.jpg
                # ./speed_limit_20/IMG_20181007_151246.jpg
                # ./speed_limit_15/IMG_20181007_145412.jpg
                if img_sz < 100000:
                    print(f"skipping {image_path}")
                    continue  # skip all images less than 100kB (corrupt) (there should only be 3)
                
                for label in labels:
                    if label.class_index in self.splits:
                        colour_image = cv.imread(image_path)
                        gray_image = np.array(cv.cvtColor(colour_image, cv.COLOR_BGR2GRAY)) # load gray image in numpy array
                        xmin = label.bounds.left
                        xmax = label.bounds.right
                        ymin = label.bounds.top
                        ymax = label.bounds.bottom
                        crop = gray_image[ymin:ymax, xmin:xmax]
                        # if need_print:
                        #     print('stage 1',crop)
                        img = cv.resize(crop,(32, 32))
                        img = cv.equalizeHist(img)
                        # if need_print:
                        #     print('stage 2', img)
                        #     need_print = False
                        self.splits[label.class_index].append(np.array(img, dtype=np.float32))
        
        for class_x, crops in self.splits.items():
            self.splits[class_x] = np.array(crops, dtype='float32')
        
        # self.save_np_arrays()

    
    def save_np_arrays(self, force: bool = False):
        print("Saving pre-processed crops...")
        for class_a, background in self.compare:
            zeros = self.splits[class_a]
            ones = None
            for class_x in background:
                if ones is None:
                    ones = self.splits[class_x]
                else:
                    np.append(ones, self.splits[class_x], axis=0)
            
            
            data_x = np.concatenate((zeros, ones), axis=0)
            labels = np.concatenate((np.zeros(len(zeros)), np.ones(len(ones)))) # class_a corresponds to 0 and so on
            labels = np.array(labels, dtype=np.int32)
            assert len(zeros) + len(ones) == len(labels)
            subfolder = os.path.join(self.path, str(self.dataset.classes[class_a]))
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            data_path = os.path.join(subfolder, "data.npy")
            labels_path = os.path.join(subfolder, "labels.npy")
            data_x = np.reshape(data_x,(data_x.shape[0],data_x.shape[1]*data_x.shape[2]))
            np.save(data_path, data_x)
            np.save(labels_path, np.array(labels, dtype=np.int32))
        
        print("Save complete.")
        print("Saved at: " + self.path)







