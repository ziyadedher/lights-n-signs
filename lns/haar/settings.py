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
    feature_type: str = "HAAR" # TODO: change to HAAR and check

    # Width and height of the features to learn
    feature_size: int = 15

    # Number of samples to generate in setup
    num_samples: int = 3000

    # Number of stages to train and respective number of positive and negative samples to use
    num_stages: int = 25
    num_positive: int = 1000
    num_negative: int = 2000

    # Minimal desired hit rate for each stage of the classifier
    min_hit_rate: float = 0.995
    max_false_alarm: float = 0.3

    # Inference metrics, how much to scale the features by neighbour threshold
    scale_factor: float = 1.1
    min_neighbours: float = 4 #3
    # establishes how 
# pylint: enable=too-many-instance-attributes
