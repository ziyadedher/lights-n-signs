from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor
from typing import List
import cv2 as cv # type: ignore
import numpy as np
import os
from tqdm import tqdm # type: ignore

class SVMProcessor:
    def __init__(self, path: str, dataset: Dataset, compare: List[tuple]):
        """Handles preprocessing of dataset

        Args:
            path (str): [path to store preprocessed dataset]
            dataset (str): [path to dataset]
            compare (List[tuple]): [List of tuples containing class indices to compare]
        """
        self.dataset = dataset 
        self.path = path
        self.compare = compare
        self.splits = {}
        for a, b in compare:
            self.splits[a] = []
            self.splits[b] = []


    def preprocess(self, force: bool = True):
        if force or not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        else:
            print("Dataset already processed")
            return

        print("Creating crops...")
        with tqdm(desc="Processing", total=len(self.dataset.annotations.keys()), miniters=1) as tqdm_bar:
            for image_path, labels in self.dataset.annotations.items():
                tqdm_bar.update()
                colour_image = cv.imread(image_path)
                gray_image = np.array(cv.cvtColor(colour_image, cv.COLOR_BGR2GRAY)) # load gray image in numpy array
                for label in labels:
                    if label.class_index in self.splits:
                        xmin = label.bounds.left
                        xmax = label.bounds.right
                        ymin = label.bounds.top
                        ymax = label.bounds.bottom
                        crop = gray_image[ymin:ymax, xmin:xmax]
                        self.splits[label.class_index].append(crop)
        # self.save_np_arrays()

    
    def save_np_arrays(self, force: bool = False):
        print("Saving pre-processed crops...")
        for class_a, class_b in self.compare:
            zeros = np.array(self.splits[class_a], dtype=object) # all crops of class_a
            ones = np.array(self.splits[class_b], dtype=object) # all crops of class_b
            data_x = np.concatenate((zeros, ones), axis=0)
            labels = np.concatenate((np.zeros(len(zeros)), np.ones(len(ones)))) # class_a corresponds to 0 and so on
            assert len(zeros) + len(ones) == len(labels)
            subfolder = os.path.join(self.path, str(self.dataset.classes[class_a]) + '_' + self.dataset.classes[class_b])
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            data_path = os.path.join(subfolder, "data.npy")
            labels_path = os.path.join(subfolder, "labels.npy")
            np.save(data_path, data_x)
            np.save(labels_path, np.array(labels, dtype=np.int64))
        
        print("Save complete.")
        print("Saved at: " + self.path)







