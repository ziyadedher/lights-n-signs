"""SVM Classifier training script.

This script manages model training and saving.
"""
from typing import Optional, Union

import numpy as np
import cv2 as cv
import os

# class SVMTrainerWrapper:
#     def __init__(self, positive_folder: str, negative_folder: str, positive_np_path: str, negative_np_path: str):




class SVMTrainer():
    """Manages the training environment.

    Contains and encapsulates all training setup and files under one namespace.
    """

    def __init__(self, data_path: Optional[str], labels_path: Optional[str], model_path: str) -> None:
        """Initialize a SVM trainer and save model at model_path.

        Sources data from the given <data>, if any.
        If <load> is set to False removes any existing trained model before training.
        """
        self.data_path = data_path
        self.labels_path = labels_path
        self.model_path = model_path
        self.train_data = None
        self.labels = None

    def setup(self) -> None:
        """Load datasets required for training.
        """
        if self.data_path and self.labels_path:
            self.train_data = np.load(self.data_path, allow_pickle=True)
            self.train_data = np.float32(self.train_data)
            print('Train Shape ',self.train_data.shape)
            print(self.train_data[0])
            #print('type',type(self.train_data[0][0][0]))
            # for i in range(self.train_data.shape[0]):
            #     self.train_data[i] = np.int32(self.train_data[i])
            
            
            self.labels = np.load(self.labels_path, allow_pickle=True)
            self.labels = np.int32(self.labels)
            self.labels = np.reshape(self.labels,(self.labels.shape[0],1))
            print('Shape ',self.labels.shape)
            # for i in range(self.labels.shape[0]):
            #     self.labels[i] = np.int32(self.labels[i])
            
            # self.labels = np.array(labels, dtype=np.float32)

            # print(type(self.train_data))
            # print(self.train_data.shape)
            # print(type(self.train_data[0][0][0]))
            # print(type(self.labels[0]))
            # print(self.labels.shape)



            if len(self.train_data) == 0:
                raise FileNotFoundError("Empty training data!")
                
        else:
            raise FileNotFoundError("Training data is not provided")

    def train(self) -> None:
        """Begin training the model.

        Train for <num_stages> stages before automatically stopping and generating the trained model.
        Train on <num_positive> positive samples and <num_negative> negative samples.
        """

        svm_model = cv.ml.SVM_create()
        svm_model.setType(cv.ml.SVM_C_SVC)
        svm_model.setKernel(cv.ml.SVM_LINEAR)
        svm_model.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

        print("\n\nTraining")
        svm_model.train(self.train_data, cv.ml.ROW_SAMPLE, self.labels)
        # svm_model.train(samples=self.train_data,
        #                 layout=cv.ml.ROW_SAMPLE, responses=self.labels)
        print("\nTraining completed")
        print(self.model_path+'/svm.xml')
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        svm_model.save(self.model_path+'/svm.xml')
        print(f"Saved model at {self.model_path}/svm.xml")
