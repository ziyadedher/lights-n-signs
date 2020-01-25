from typing import Optional, Tuple

import os
import sys

import cv2  # type: ignore

from lns.common.model import Model
from lns.common.dataset import Dataset
from lns.common.structs import Object2D

def visualize_image(image_path: str, *,
                    model: Optional[Model] = None, visualize_model: bool = False,
                    labels: Optional[Dataset.Labels] = None, show_labels: bool = False) -> None:
    image = cv2.imread(image_path)

    if show_labels:
        if labels is None:
            raise ValueError("Labels cannot be none if <show_labels> is set to `True`.")
        image = put_labels_on_image(image, labels)

    if visualize_model:
        if model is None:
            raise ValueError("Need to set a model if <visualize_model> is Optional[] set to `True`.")
 #       image = put_predictions_on_image(image, model.predict(image))
    handle_image_window(image)

def handle_image_window(self, image: np.ndarray) -> None:
    cv2.imshow("visualization", image)
    key = cv2.waitKey(0)
    while key != ord("a"):
        key = cv2.waitKey(0)
        if key == 27:
            sys.exit(0)

def generate_video_stream(annotations: Dict[str, Object2D], *, 
                        output_path: Optional[str] = os.path.curdir() + 'output.mp4', fps: Optional[int] = 5,
                        size: Optional[Tuple[int, int]] = (1920, 1080), num_frames: Optional[int] = 1000) -> None:
    frame_stream = []
    frame_count = 0
    for image_path in annotations:
        if frame_count < num_frames:
            frame_stream.append(visualize_image(image_path, model=None, visualize_model=False, labels=annotations[image_path], show_labels=True))
            frame_count += 1
        else:
            break
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for frame in frame_stream:
        video_writer.write(frame)
    video_writer.release()

def put_labels_on_image(image: np.ndarray, labels: Dataset.Labels) -> np.ndarray:
    for label in labels:
        image = cv2.rectangle(image, (label.bounds.left, label.bounds.top), (label.bounds.right, label.bounds.bottom), (0, 255, 0))
    return image

if __name__ == '__main__':
    cv2.namedWindow("visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("visualization", 1920, 1080)

    from lns.common.preprocess import Preprocessor
    bosch = Preprocessor.preprocess("Bosch")
    #lights = Preprocessor.preprocess("lights")
    #scale = Preprocessor.preprocess("scale_lights")pass
    # print(Preprocessor._dataset_preprocessors)
    # mturk = Preprocessor.preprocess("scale_lights")

    dataset = dataset.merge_classes({
        "green": [
            "GreenLeft", "Green", "GreenRight", "GreenStraight",
            "GreenStraightRight", "GreenStraightLeft", "Green traffic light"
        ],
        "red": [
            "Yellow", "RedLeft", "Red", "RedRight", "RedStraight",
            "RedStraightLeft", "Red traffic light", "Yellow traffic light"
        ],
        "off": ["off"]
    })

    dataset = bosch
    #from lns.squeezedet.model import SqueezeDetModel
    #model = SqueezeDetModel("/home/lns/lns/xiyan/models/alllights-414000/train/model.ckpt-415500")

    annotations = dataset.annotations
    # for image_path in annotations:
    #     visualize_image(image_path, model=None, visualize_model=False, labels=annotations[image_path], show_labels=True)

    generate_video_stream(annotations)
