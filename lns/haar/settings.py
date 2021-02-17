"""Manages settings related to Haar training and evaluation."""

from dataclasses import dataclass

from lns.common.settings import Settings


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class HaarSettings(Settings):
    """Settings encapsulation for all Haar trainer settings."""

    # Index of the class in the dataset used
    class_index: int = 0

    # Features to use. Can be HAAR or LBP
    feature_type: str = "LBP" # TODO: change to HAAR and check

    # Width and height of the features to learn
    feature_size: int = 10

    # Number of samples to generate in setup
    num_samples: int = 1500

    # Number of stages to train and respective number of positive and negative samples to use
    num_stages: int = 30
    num_positive: int = 1000
    num_negative: int = 500

    # Minimal desired hit rate for each stage of the classifier
    min_hit_rate: float = 0.995

    # Inference metrics, how much to scale the features by neighbour threshold
    scale_factor: float = 1.1
    min_neighbours: float = 3
# pylint: enable=too-many-instance-attributes
