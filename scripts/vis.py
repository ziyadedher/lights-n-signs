import cv2

from lns.common.preprocess import Preprocessor
from lns.common.visualization import visualize_image
# from lns.yolo.train import YoloTrainer


dataset = Preprocessor.preprocess("nuscenes")
# dataset = dataset.merge_classes({
#     "green": ["goLeft", "Green", "GreenLeft", "GreenStraightRight", "go", "GreenStraightLeft", "GreenRight", "GreenStraight", "3-green", "4-green", "5-green"],
#     "yellow": ["warning", "Yellow", "warningLeft", "3-yellow", "4-yellow", "5-yellow"],
#     "red": ["stop", "stopLeft", "RedStraightLeft", "Red", "RedLeft", "RedStraight", "RedRight", "3-red", "4-red", "5-red"],
#     "off": ["OFF", "off", "3-off", "3-other", "4-off", "4-other", "5-off", "5-other"]
# })

# trainer = YoloTrainer("new_dataset_ac_21")
print(dataset.classes)
for image in dataset.images:
    img = visualize_image(image, labels=dataset.annotations[image], classes=dataset.classes, show_labels=True)
    cv2.imshow("image", img)
    cv2.waitKey(0)
# visualize(dataset, show_truth=True)
