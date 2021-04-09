"""Module to visualize any model on a given dataset.

Given a Trainer object and a Dataset object, the module
is capable of either visualizing the labels frame-by-frame, or
by generating a video stream.
"""

from typing import Optional, Tuple, Dict

import cv2  # type: ignore
import numpy as np  # type: ignore
import os
from os.path import isfile, join
from lns.common.model import Model
from lns.common.train import Trainer
from lns.common.dataset import Dataset


def visualize_image(image_path: str, *,
                    model: Optional[Model] = None, visualize_model: bool = False, threshold: float = 0.2,
                    labels: Optional[Dataset.Labels] = None, classes: Dataset.Classes,
                    show_labels: bool = False, color_mapping: Optional[Dict] = None,crop = None) -> np.ndarray:
    """Visualizes the predictions of any model on a single frame in the dataset."""
    image = cv2.imread(image_path)
    imaged_cropped = image
    if crop:
        h,w,channels = image.shape
        x_min = crop[0]
        x_max = crop[1]
        y_min = crop[2]
        y_max = crop[3]
        
        x_min_pixels = int(x_min*w)
        x_max_pixels = int(x_max*w)
        y_min_pixels = int(y_min*h)
        y_max_pixels = int(y_max*h)
        image_cropped = image[y_min_pixels:y_max_pixels,x_min_pixels:x_max_pixels]

    if show_labels:
        if labels is None or classes is None:
            raise ValueError("Labels cannot be none if <show_labels> is set to `True`.")
        image = _put_labels_on_image(image, labels, classes)

    if visualize_model:
        if model is None:
            raise ValueError("Need to set a trainer if <visualize_model> is Optional[] set to `True`.")
        image = _put_labels_on_image(image, model.predict(image_cropped), classes, is_pred=True, color_mapping=color_mapping,
                                     threshold=threshold,crop = crop)
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
                         color_mapping: Optional[Dict] = None, threshold: float = 0.2, crop = None) -> np.ndarray:
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
    h,w,channels = image.shape

    x_min_pixels = 0
    x_max_pixels = w
    y_min_pixels = 0
    y_max_pixels = h
    if crop:
        
        x_min = crop[0]
        x_max = crop[1]
        y_min = crop[2]
        y_max = crop[3]
        
        x_min_pixels = int(x_min*w)
        x_max_pixels = int(x_max*w)
        y_min_pixels = int(y_min*h)
        y_max_pixels = int(y_max*h)
        image_cropped = image[y_min_pixels:y_max_pixels,x_min_pixels:x_max_pixels]

    for label in labels:
        if label.score > threshold:
            if not color_mapping:
                lbl = classes[label.class_index]
            else:
                lbl = color_mapping.get(classes[label.class_index], "red")
            
            image = cv2.rectangle(
                image,
                (x_min_pixels, y_min_pixels),
                (x_max_pixels, y_max_pixels),
                (255,255,255),3
            )
            image = cv2.rectangle(
                image,
                (int(label.bounds.left)+x_min_pixels, int(label.bounds.top)+y_min_pixels),
                (int(label.bounds.right)+x_min_pixels, int(label.bounds.bottom)+y_min_pixels),
                class_to_color[lbl]
            )
            label_score = f'{label.score:.2f}' if label.score != 1 else ''
            image = cv2.putText(
                image,
                f'{classes[label.class_index]} {label_score}',
                (int(label.bounds.right)+x_min_pixels, int(label.bounds.bottom)+y_min_pixels),
                cv2.FONT_HERSHEY_PLAIN, 3,
                class_to_color[lbl], thickness=2
            )
    return image

def generate_video_stream_from_images(folder_path: str, *,
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
                                                threshold=threshold,
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

def test_single_image(test_image_path, destination_path, model): #this is mostly a dummy function showing one way to run inference on a single image

    from lns.yolo import YoloTrainer

    # TRAINER = YoloTrainer('pedestrian_5')


    # generate_video_stream(DATASET, trainer=TRAINER)
    image = cv2.imread(test_image_path)
    img_w_labels = visualize_image(test_image_path, model=model,visualize_model=True,classes = ['Pedestrian'])
    cv2.imwrite(destination_path,img_w_labels)

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames

def convert_video_to_frames(video_path: str, output_folder_path: str, frame_rate: Optional[float] = 0.5,crop = None):
    vidcap = cv2.VideoCapture(video_path)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if hasFrames:
            if crop:
                x_min = crop[0]
                x_max = crop[1]
                y_min = crop[2]
                y_max = crop[3]
                
                x_min_pixels = int(x_min*w)
                x_max_pixels = int(x_max*w)
                y_min_pixels = int(y_min*h)
                y_max_pixels = int(y_max*h)
                image = image[y_min_pixels:y_max_pixels,x_min_pixels:x_max_pixels]
            cv2.imwrite(output_folder_path + "/image"+str(count)+".jpg", image)     # save frame as JPG file
        return hasFrames
    sec = 0
    #frameRate = 0.5 #//it will capture image in each 0.5 second
    count=1
    success = getFrame(sec)
    while success: #capture a frame every frame_rate seconds
        count = count + 1
        sec = sec + 1/frame_rate
        sec = round(sec, 2)
        success = getFrame(sec)
    return True

def label_video_from_file(frame_folder_path: str, *,
                          output_path: Optional[str] = 'output.avi', fps: Optional[int] = 5,
                          size: Optional[Tuple[int, int]] = (1920, 1080),
                          model: Optional[Model] = None, num_frames: int = 1000,
                          trainer_color_mapping: Optional[Dict] = None, threshold: float = 0.01, pedestrian = False, crop = False) -> None:
    """Generate a video stream with the predictions of the model drawn onto each frame in the video file."""
    frame_stream = []
    frame_count = 0
    image_paths = [f for f in os.listdir(frame_folder_path) if isfile(join(frame_folder_path,f))]

    image_paths.sort(key = lambda x: int(x[5:-4]))

    #image_paths.sort()

    print('Writing video stream to:', output_path)
    if model:
        pass
    else:
        raise ValueError("You did not pass a model")

    # Determine if it is inference for pedestrians or lights
    if pedestrian:
        class_label = ["pedestrians"]
    else:
        # class_label = ['5-red-yellow','yellow','green','off','5-red-green','4-red-green','red']
        class_label = ['off', '5-red-green','4-red-green','green','yellow','5-red-yellow','red']
        trainer_color_mapping = dict(zip(['5-red-yellow','yellow','green','off','5-red-green','4-red-green','red'],
                             ['red'] + ['yellow'] + ['green'] + ['off'] + ['red'] * 3 ))
        '''
    
        class_label = ['yellow','5-red-green','red', '4-red-green', 'off', '5-red-yellow','green']
        trainer_color_mapping = dict(zip(['yellow','5-red-green','red', '4-red-green', 'off', '5-red-yellow','green'],
                             ['yellow'] + ['red'] * 3+ ['off']+ ['red']+['green']))

        
        class_label = ['4-red-green', '5-red-green', '5-red-yellow','green','off','red','yellow']
        trainer_color_mapping = dict(zip(['4-red-green', '5-red-green', '5-red-yellow','green','off','red','yellow'],
                             ['red'] * 3 + ['green'] + ['off']+ ['red']+['yellow']))
    
        
        class_label = ['5-red-yellow','yellow','green','off','5-red-green','4-red-green','red']
        trainer_color_mapping = dict(zip(['5-red-yellow','yellow','green','off','5-red-green','4-red-green','red'],
                             ['red'] + ['yellow'] + ['green'] + ['off'] + ['red'] * 3 ))
        
        class_label = ['5-red-green', '4-red-green', 'red', '5-red-yellow', 'green', 'yellow', 'off']
        trainer_color_mapping = dict(zip(['5-red-green', '4-red-green', 'red', '5-red-yellow', 'green', 'yellow', 'off'],
                             ['red'] * 4 + ['green'] + ['yellow'] + ['off']))
        '''

    for image_path in image_paths:
        if frame_count < num_frames:
            frame_stream.append(visualize_image(os.path.join(frame_folder_path,image_path),
                                                model=model, visualize_model=True,
                                                threshold=threshold,
                                                classes=class_label, color_mapping=trainer_color_mapping,crop = crop))
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
