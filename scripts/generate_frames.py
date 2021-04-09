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
def gen_frames(fps=5,GPU=0):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU) #choose which GPU to use, I don't actually know if this code will use GPU


    # Video file structure is as follows: ../ped_dummy_vides_2021/timestamped folder/videos
    # Each timestamped folder contains 4 videos, 1 from each camera on Zeus.
    video_parent_path = "/mnt/ssd2/od/datasets/lights_vids/"
    frames_folder_parent = "/mnt/ssd2/od/datasets/lights_vids/frames_cropped"


    if gen_frames:
        print("Generating image frames from video file")
        for video_folder in os.listdir(video_parent_path):
            if video_folder in ['traffic_control','intersection_challenge']:
                for video_file in os.listdir(os.path.join(video_parent_path,video_folder)):
                    output_folder_path = os.path.join(frames_folder_parent,video_folder,video_file[:-4])
                    Path(output_folder_path).mkdir(parents=True,exist_ok = True)
                    convert_video_to_frames(os.path.join(video_parent_path, video_folder, video_file),output_folder_path,frame_rate = fps,crop = [0.25,0.75,0.175,0.5])

if __name__ == '__main__':
    fps = 5 #how many frames per second to generate frames at
    GPU = 1 #which GPU to use
    gen_frames(fps,GPU)