"""Test the anchor generation and compare to YOLO."""
from lns.common.preprocess import Preprocessor
dataset = Preprocessor.preprocess('ScaleLights')
anchors = dataset.generate_anchors(9)


# print(anchors)
