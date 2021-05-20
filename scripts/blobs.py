from pathlib import Path

import cv2
import tensorflow as tf

from lns.yolo.train import YoloTrainer
from lns.common import visualization


TRAINER_NAME = "new_dataset_ac_21"
IMAGE_PATH = Path("input2.png").resolve()


trainer = YoloTrainer(TRAINER_NAME)
model = trainer.model
classes = trainer.dataset.classes

img = cv2.imread(str(IMAGE_PATH))
prediction = model.predict(img)
img = visualization.put_labels_on_image(img, prediction, classes, is_pred=True)

cv2.imshow("output", img)
cv2.waitKey(0)
