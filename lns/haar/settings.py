"""Manages settings related to Haar training and evaluation."""

from dataclasses import dataclass

from lns.common.settings import Settings


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class HaarSettings(Settings):
    """Settings encapsulation for all Haar trainer settings."""

    # Index of the class in the dataset used
    class_index: int = 0 # 0 to 3
    # class to classify = ['nrt_nlt_sym', 'nrt_nlt_text', 'Stop', 'Yield'][class_index] 

    # Features to use. Can be HAAR or LBP
    feature_type: str = "LBP" # "HAAR"

    # Width and height of the features to learn
    feature_size: int = 12

    # Number of samples to generate in setup
    num_samples: int = 4300

    # Number of stages to train and respective number of positive and negative samples to use
    num_stages: int = 16
    num_positive: int = 1200
    num_negative: int = 6000

    # Minimal desired hit rate for each stage of the classifier
    min_hit_rate: float = 0.995
    max_false_alarm: float = 0.3

    # Inference metrics, how much to scale the features by neighbour threshold
    scale_factor: float = 1.06
    min_neighbours: float = 2
# pylint: enable=too-many-instance-attributes
