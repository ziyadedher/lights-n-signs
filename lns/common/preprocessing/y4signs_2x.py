import os
from collections import Counter
from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor
from lns.common.dataset import Dataset 
import pickle

class Y4Signs_2072_1168:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def getDataset(self, path: str):
        if not os.path.isdir(path):
            raise FileNotFoundError("Could not find Y4Signs dataset on path: " + path)
        
        images: Dataset.Images = []
        classes: Dataset.Classes = []
        annotations: Dataset.Annotations = {}

        for data_path in os.listdir(path):
            if 'png' in data_path:
                image_path = os.path.join(path, data_path)
                images.append(image_path)
                annot_path = os.path.join(path, data_path[:-len('png')] + 'pkl')
                annots = None
                with open(annot_path,'rb') as annot_file:
                    annots = pickle.load(annot_file)
                annotations[image_path] = annots
        
        classes = None
        
        with open(os.path.join(path, 'classes.pkl'),'rb') as annot_file:
            classes = pickle.load(annot_file)
        
        
        return Dataset(self.dataset_name, images, classes, annotations)
                
                