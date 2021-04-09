from typing import Optional, Tuple, Dict

import os
from pathlib import Path
import cv2  # type: ignore
import numpy as np  # type: ignore

from lns.common.model import Model
from lns.common.train import Trainer
from lns.common.dataset import Dataset
from lns.common.visualization import *

from lns.yolo.train import YoloTrainer
#from lns.yolo.settings import YoloSettings

fps = 5
thresh = 0.1

os.environ["CUDA_VISIBLE_DEVICES"] = "2" #choose which GPU to use, I don't actually know if this code will use GPU


# Video file structure is as follows: ../ped_dummy_vides_2021/timestamped folder/videos
# Each timestamped folder contains 4 videos, 1 from each camera on Zeus.
frames_folder_parent = '/mnt/ssd2/od/datasets/lights_vids/frames' #"/home/od/files/datasets/ped_dummy_vids_2021/frames"
labelled_videos_parent = "/home/lns/helen/lights-n-signs-training/labelled_videos" # "/home/od/files/datasets/ped_dummy_vids_2021/labelled_videos"

trainer_name = 'matthieu_darknet53_416_3' #'new_dataset_ac_small_3' #'new_dataset_ac_medium_1' #'smaller_800x640_1' #'new_dataset_ac_small_3' #'new_dataset_ac_1' # "yolo_ped_mbd_trial_29"
trainer = YoloTrainer(trainer_name)
model = trainer.model

folder_of_interest = 'traffic_control'



print("Generating labelled video from image frames")
for frames_folder in os.listdir(frames_folder_parent): #iterate through each dated folder, or groups of 4 videos that were taken simultaneously
    #if frames_folder == folder_of_interest:
    output_folder_path = os.path.join(labelled_videos_parent,trainer_name,frames_folder)
    Path(output_folder_path).mkdir(parents=True,exist_ok=True)
    for video_frames in os.listdir(os.path.join(frames_folder_parent,frames_folder)): #iterate through each of the 4 camera videos on the specific date
        
        frames_path = os.path.join(frames_folder_parent,frames_folder,video_frames)
        label_video_from_file(frames_path ,output_path = os.path.join(output_folder_path, video_frames+ "_" +trainer_name + "_inference.avi"),fps=fps,model=model,threshold=thresh,crop = [0.25,0.75,0.175,0.5])