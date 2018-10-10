"""Haar model evaluator.

This module contains scripts which generate Haar trained models
"""
from typing import Tuple, Optional
from haar.model import HaarModel
from haar.train import Trainer
from common import preprocess
from common.preprocess.preprocessing import Dataset


def evaluate(dataset_name: str, feature_size: int, num_samples: int,
             light_type: str, num_stages: int, num_positive: int,
             num_negative: int) -> Tuple[Dataset, Optional[HaarModel]]:
    """Load dataset and train model

    Train haar cascade on images from <dataset> with params
    """
    dataset = preprocess.preprocess.\
        Preprocessor.preprocess(dataset_name)
    trainer = Trainer("trainer", dataset)
    trainer.setup_training(feature_size, num_samples, light_type)
    trainer.train(num_stages, num_positive, num_negative)
    model = trainer.generate_model()
    return (dataset, model)
