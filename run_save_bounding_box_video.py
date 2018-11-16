from common.benchmarking.vidstream import save_bounding_box_video
from yolov3 import yolo


if __name__ == "__main__":
    model = yolo.YOLO()
    save_bounding_box_video(model, "out.mp4", 1920, 1440)