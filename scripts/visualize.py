from typing import Optional, Tuple, Dict

import os
import sys

import cv2  # type: ignore
import numpy as np

from lns.common.model import Model
from lns.common.dataset import Dataset
from lns.common.structs import Object2D

def visualize_image(image_path: str, *,
                    model: Optional[Model] = None, visualize_model: bool = False,
                    labels: Optional[Dataset.Labels] = None, classes: Optional[Dataset.Classes] = None,
                    show_labels: bool = False) -> np.ndarray:
    image = cv2.imread(image_path)

    if show_labels:
        if labels is None or classes is None:
            raise ValueError("Labels cannot be none if <show_labels> is set to `True`.")
        image = put_labels_on_image(image, labels, classes)

    if visualize_model:
        if model is None:
            raise ValueError("Need to set a model if <visualize_model> is Optional[] set to `True`.")
 #       image = put_predictions_on_image(image, model.predict(image))
    return image

def handle_image_window(self, image: np.ndarray) -> None:
    cv2.namedWindow("visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("visualization", 1920, 1080)
    cv2.imshow("visualization", image)
    key = cv2.waitKey(0)
    while key != ord("a"):
        key = cv2.waitKey(0)
        if key == 27:
            sys.exit(0)

def generate_video_stream(dataset: Dataset, *, 
                        output_path: Optional[str] = 'output.mp4', fps: Optional[int] = 5,
                        size: Optional[Tuple[int, int]] = (1920, 1080), num_frames: Optional[int] = 1000) -> None:
    frame_stream = []
    frame_count = 0
    annotations = dataset.annotations

    print('Writing video stream to:', output_path)
    for image_path in annotations:
        if frame_count < num_frames:
            frame_stream.append(visualize_image(image_path,
                                                model=None, visualize_model=False,
                                                labels=annotations[image_path],
                                                show_labels=True,
                                                classes=dataset.classes))
            frame_count += 1
        else:
            break
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    for frame in frame_stream:
        video_writer.write(cv2.resize(frame, size))

    print("Processed all image frames and annotated them")
    print(f"Processed {frame_count} frames")
    video_writer.release()
    print("Video stream written!")

def put_labels_on_image(image: np.ndarray, labels: Dataset.Labels, classes: Dataset.Classes, is_pred: bool = False) -> np.ndarray:
    shade = 255 if not is_pred else 150
    class_to_color = {
        'green': (0, shade, 0),
        'red': (0, 0, shade),
        'off': (0, 0, 0)
    }
    for label in labels:
        image = cv2.rectangle(image, (label.bounds.left, label.bounds.top),
                             (label.bounds.right, label.bounds.bottom),
                             classes_to_color[classes[label.class_index]])
        image = cv2.putText(image, # Put label on the image
                            classes[label.class_index],
                            (label.bounds.right, label.bounds.bottom), cv2.FONT_HERSHEY_PLAIN,
                            1, class_to_color[classes[label.class_index]], thickness=2)
    return image

if __name__ == '__main__':
    from lns.common.preprocess import Preprocessor
    bosch = Preprocessor.preprocess("Bosch")
    
    dataset = bosch
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

    # for image_path in annotations:
    #     visualize_image(image_path, model=None, visualize_model=False, labels=annotations[image_path], show_labels=True)

    generate_video_stream(dataset)
