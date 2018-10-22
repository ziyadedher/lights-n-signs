"""Haar model evaluator.

This module contains scripts which generate Haar trained models
"""
from typing import List
import numpy as np
from lns_common.test_new_preprocessing import benchmark_model
from lns_common.preprocess.preprocess import Preprocessor
from lns_common.model import Model
from lns_haar.model import Bounds2D, PredictedObject2D
from lns_haar.train import HaarTrainer
from preprocessing.artificial import SyntheticDataset
from lns_common import config
import os


class DummyModel(Model):
    def predict(self, img: np.array) -> List[PredictedObject2D]:
        return [PredictedObject2D(Bounds2D(0, 0, 100, 100), ["go"]),
                PredictedObject2D(Bounds2D(0, 0, 100, 100), ["go"]),
                PredictedObject2D(Bounds2D(0, 0, 100, 100), ["go"])]


def evaluate(dataset_name: str) -> None:
    """Load dataset and train model.

    Train haar cascade on images from <dataset> with params
    """
    dataset = Preprocessor.preprocess(dataset_name)
    trainer = HaarTrainer("trainer", dataset)
    trainer.setup_haar(24, 5000, "go")
    trainer.train_haar(100, 4000, 2000)
    model = trainer.generate_model()
    benchmark_model(dataset, model)


def evaluate_synthetic(path: str) -> None:
    """Test the creation and use of a synthetic dataset."""
    dataset = SyntheticDataset("synthetic", path, ['donotenter'])
    trainer = HaarTrainer("synthetic_trainer", dataset)
    trainer.setup_haar(24, 2000, "donotenter")
    trainer.train_haar(30, 1500, 700)
    model = trainer.generate_model()
    # benchmark_model(dataset, model)


if __name__ == '__main__':
    # evaluate("LISA")
    evaluate_synthetic(
        os.path.join(
            config.RESOURCES_ROOT,
            "haar/data/sign/sign_samples"
        )
    )
