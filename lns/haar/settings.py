"""Manages settings related to Haar training and evaluation."""

from dataclasses import dataclass

from lns.common.settings import Settings


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class HaarSettings(Settings):
    """Settings encapsulation for all Haar trainer settings."""

    # Index of the class in the dataset used
    class_index: int = 1 # 0 to 3
    # class to classify = ['nrt_nlt_sym', 'nrt_nlt_rto_lto_text', 'Stop', 'Yield'][class_index] 

    # Features to use. Can be HAAR or LBP
    feature_type: str = "LBP"

    # Width and height of the features to learn
    width: int = 24
    height: int = 32

    # Number of samples to generate in setup
    num_samples: int = 7000

    # Number of stages to train and respective number of positive and negative samples to use
    num_stages: int = 16
    num_positive: int = 5000
    num_negative: int = 8000

    # Minimal desired hit rate for each stage of the classifier
    min_hit_rate: float = 0.996217
    max_false_alarm: float = 0.3908

    # Inference metrics, how much to scale the features by neighbour threshold
    scale_factor: float = 1.08
    min_neighbours: float = 0
# pylint: enable=too-many-instance-attributes
