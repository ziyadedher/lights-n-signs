import numpy as np

from lns.common.preprocess import Preprocessor
from lns.yolo.train import YoloTrainer
from lns.common import evaluation

trainer = YoloTrainer("new_dataset_ac_21")
dataset = Preprocessor.preprocess("ScaleLights_New_Youtube")
# dataset = dataset.merge_classes({
#     "green": ["goLeft", "Green", "GreenLeft", "GreenStraightRight", "go", "GreenStraightLeft", "GreenRight", "GreenStraight", "3-green", "4-green", "5-green"],
#     "yellow": ["warning", "Yellow", "warningLeft", "3-yellow", "4-yellow", "5-yellow"],
#     "red": ["stop", "stopLeft", "RedStraightLeft", "Red", "RedLeft", "RedStraight", "RedRight", "3-red", "4-red", "5-red", "4-red-green", "5-red-green", "5-red-yellow"],
#     "off": ["OFF", "off", "3-off", "3-other", "4-off", "4-other", "5-off", "5-other"]
# })
dataset = dataset.merge_classes({
  "green": ["goLeft", "Green", "GreenLeft", "GreenStraightRight", "go", "GreenStraightLeft", "GreenRight", "GreenStraight", "3-green", "4-green", "5-green"],
  "yellow": ["warning", "Yellow", "warningLeft", "3-yellow", "4-yellow", "5-yellow"],
  "red": ["stop", "stopLeft", "RedStraightLeft", "Red", "RedLeft", "RedStraight", "RedRight", "3-red", "4-red", "5-red"],
  "off": ["OFF", "off", "3-off", "3-other", "4-off", "4-other", "5-off", "5-other"]
})
print(dataset.classes)
print(trainer.dataset.classes)

class_mapping = {trainer.dataset.classes.index(name): dataset.classes.index(name) for name in dataset.classes}
print(class_mapping)

model = trainer.model
class_mapping_fn = lambda x: class_mapping[x]

mats = evaluation.confusion(
    model, dataset,
    class_mapping=class_mapping_fn,
    num_to_sample=2000,
    iou_threshold=0.05,
    score_threshold=np.linspace(0., 1., 21),
)

metrics = []
for mat in mats:
    metrics.append(evaluation.metrics(mat))

np.save("output.npy", metrics)

# latency = evaluation.latency(trainer.model, dataset, num_to_sample=100)
# print(np.median(latency))
