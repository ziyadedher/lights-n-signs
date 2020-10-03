"""Module to visualize any model on a given dataset.

Given a Trainer object and a Dataset object, the module
is capable of either visualizing the labels frame-by-frame, or
by generating a video stream.
"""

from typing import Optional, Tuple, Dict

import cv2  # type: ignore
import numpy as np  # type: ignore

from lns.common.model import Model
from lns.common.train import Trainer
from lns.common.dataset import Dataset


def visualize_image(image_path: str, *,
                    model: Optional[Model] = None, visualize_model: bool = False, threshold: float = 0.2,
                    labels: Optional[Dataset.Labels] = None, classes: Dataset.Classes,
                    show_labels: bool = False, color_mapping: Optional[Dict] = None) -> np.ndarray:
    """Visualizes the predictions of any model on a single frame in the dataset."""
    image = cv2.imread(image_path)

    if show_labels:
        if labels is None or classes is None:
            raise ValueError("Labels cannot be none if <show_labels> is set to `True`.")
        image = _put_labels_on_image(image, labels, classes)

    if visualize_model:
        if model is None:
            raise ValueError("Need to set a trainer if <visualize_model> is Optional[] set to `True`.")
        image = _put_labels_on_image(image, model.predict(image), classes, is_pred=True, color_mapping=color_mapping,
                                     threshold=threshold)
    return image


def generate_video_stream(dataset: Dataset, *,
                          output_path: Optional[str] = 'output.avi', fps: Optional[int] = 5,
                          size: Optional[Tuple[int, int]] = (1920, 1080),
                          trainer: Optional[Trainer] = None, num_frames: int = 1000,
                          trainer_color_mapping: Optional[Dict] = None, threshold: float = 0.2) -> None:
    """Generate a video stream with the predictions of the model drawn onto each frame in the dataset."""
    frame_stream = []
    frame_count = 0
    annotations = dataset.annotations

    print('Writing video stream to:', output_path)
    if trainer:
        model = trainer.model
    else:
        raise ValueError("You did not pass a trainer")

    for image_path in annotations:
        if frame_count < num_frames:
            frame_stream.append(visualize_image(image_path,
                                                model=model, visualize_model=True,
                                                labels=annotations[image_path], threshold=threshold,
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


# pylint: disable=too-many-arguments
def _put_labels_on_image(image: np.ndarray, labels: Dataset.Labels, classes: Dataset.Classes, is_pred: bool = False,
                         color_mapping: Optional[Dict] = None, threshold: float = 0.2) -> np.ndarray:
    shade = 255 if not is_pred else 150
    class_to_color = {
        **{cls: (255, 255, 255) for cls in classes},
        **{
            'green': (0, shade, 0),
            'red': (0, 0, shade),
            'yellow': (0, 153, shade),
            'off': (0, 0, 0)
        },
    }
    for label in labels:
        if label.score > threshold:
            if not color_mapping:
                lbl = classes[label.class_index]
            else:
                lbl = color_mapping.get(classes[label.class_index], "red")

            image = cv2.rectangle(
                image,
                (int(label.bounds.left), int(label.bounds.top)),
                (int(label.bounds.right), int(label.bounds.bottom)),
                class_to_color[lbl]
            )
            label_score = f'{label.score:.2f}' if label.score != 1 else ''
            image = cv2.putText(
                image,
                f'{classes[label.class_index]} {label_score}',
                (int(label.bounds.right), int(label.bounds.bottom)),
                cv2.FONT_HERSHEY_PLAIN, 1,
                class_to_color[lbl], thickness=2
            )
    return image


if __name__ == '__main__':
    from lns.common.preprocess import Preprocessor
    BOSCH = Preprocessor.preprocess("Bosch")
    DATASET = BOSCH
    DATASET = DATASET.merge_classes({
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
    TRAINER = YoloTrainer('new_dataset_ac_1')
    # pred_class: any("red", "green", "yellow", "off") # This for coloring only
    COLOR_MAPPING = dict(zip(['5-red-green', '4-red-green', 'red', '5-red-yellow', 'green', 'yellow', 'off'],
                             ['red'] * 4 + ['green'] + ['yellow'] + ['off']))

    generate_video_stream(DATASET, trainer=TRAINER, trainer_color_mapping=COLOR_MAPPING)
