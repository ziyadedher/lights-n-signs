from typing import Optional, Tuple, Dict

import os
import sys

import cv2  # type: ignore
import numpy as np

from lns.common.model import Model
from lns.common.train import Trainer
from lns.common.dataset import Dataset
from lns.common.structs import Object2D

def visualize_image(image_path: str, *,
                    model: Optional[Model] = None, visualize_model: bool = False,
                    labels: Optional[Dataset.Labels] = None, classes: Optional[Dataset.Classes] = None,
                    show_labels: bool = False, color_mapping: Optional[Dict] = None) -> np.ndarray:
    image = cv2.imread(image_path)

    if show_labels:
        if labels is None or classes is None:
            raise ValueError("Labels cannot be none if <show_labels> is set to `True`.")
        image = put_labels_on_image(image, labels, classes)

    if visualize_model:
        if model is None:
            raise ValueError("Need to set a trainer if <visualize_model> is Optional[] set to `True`.")
        image = put_labels_on_image(image, model.predict(image), trainer.dataset.classes, is_pred=True, color_mapping=color_mapping)
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
                        output_path: Optional[str] = 'output.avi', fps: Optional[int] = 5,
                        size: Optional[Tuple[int, int]] = (1920, 1080),
                        trainer: Optional[Trainer] = None, num_frames: Optional[int] = 1000,
                        trainer_color_mapping: Optional[Dict] = None) -> None:
    frame_stream = []
    frame_count = 0
    annotations = dataset.annotations

    print('Writing video stream to:', output_path)
    model = trainer.model
    for image_path in annotations:
        if frame_count < num_frames:
            frame_stream.append(visualize_image(image_path,
                                                model=model, visualize_model=True,
                                                labels=annotations[image_path],
                                                show_labels=True,
                                                classes=dataset.classes, color_mapping=trainer_color_mapping))
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

def put_labels_on_image(image: np.ndarray, labels: Dataset.Labels, classes: Dataset.Classes, is_pred: bool = False,
                        color_mapping: Optional[Dict] = None) -> np.ndarray:
    shade = 255 if not is_pred else 150
    class_to_color = {
        'green': (0, shade, 0),
        'red': (0, 0, shade),
        'yellow': (shade, 153, 0), 
        'off': (0, 0, 0)
    }
    for label in labels:
        lbl = classes[label.class_index] if not color_mapping else color_mapping.get(classes[label.class_index], "red")
        image = cv2.rectangle(image, (label.bounds.left, label.bounds.top),
                             (label.bounds.right, label.bounds.bottom),
                             class_to_color[lbl])
        image = cv2.putText(image, # Put label on the image
                            classes[label.class_index],
                            (label.bounds.right, label.bounds.bottom), cv2.FONT_HERSHEY_PLAIN,
                            1, class_to_color[lbl], thickness=2)
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

    from lns.yolo import YoloTrainer
    trainer = YoloTrainer('new_dataset_ac_1')
    color_mapping = {
        #pred_class: any("red", "green", "yellow", "off") # This for coloring only
        k : v for k, v in zip(['5-red-green', '4-red-green', 'red', '5-red-yellow', 'green', 'yellow', 'off'],
                                ['red'] * 4 + ['green'] + ['yellow'] + ['off'])
    }
    generate_video_stream(dataset, trainer=trainer, trainer_color_mapping=color_mapping)