import numpy as np

from lns.common.preprocess import Preprocessor
from lns.yolo.train import YoloTrainer
from lns.common import evaluation

# from lns.squeezedet.train import SqueezedetTrainer
# trainer = SqueezedetTrainer('squeezedet_fullres_tiffany_eval_epoch5')
# trainer = SqueezedetTrainer('first-real_copy1')

trainer = YoloTrainer("darknet25_640_416_helen")
# trainer = YoloTrainer("darknet25_fullres_fullset_helen_1")

dataset_scale = Preprocessor.preprocess('ScaleLights')
dataset_utias = Preprocessor.preprocess('ScaleLights_New_Utias')
dataset_youtube = Preprocessor.preprocess('ScaleLights_New_Youtube')
dataset_all = dataset_scale + dataset_utias + dataset_youtube

dataset_all = dataset_all.merge_classes({
  "green": ["goLeft", "Green", "GreenLeft", "GreenStraightRight", "go", "GreenStraightLeft", "GreenRight", "GreenStraight", "3-green", "4-green", "5-green"],
  "yellow": ["warning", "Yellow", "warningLeft", "3-yellow", "4-yellow", "5-yellow"],
  "red": ["stop", "stopLeft", "RedStraightLeft", "Red", "RedLeft", "RedStraight", "RedRight", "3-red", "4-red", "5-red"],
  "off": ["OFF", "off", "3-off", "3-other", "4-off", "4-other", "5-off", "5-other"]
})
print(f"dataset_all.classes: {dataset_all.classes}")
print(f"trainer.dataset.classes: {trainer.dataset.classes}")

class_mapping = {trainer.dataset.classes.index(name): dataset_all.classes.index(name) for name in dataset_all.classes}
print(f"class_mapping: {class_mapping}")

model = trainer.model # Causing eror
class_mapping_fn = lambda x: class_mapping[x]
print("after model")

mats = evaluation.confusion(
    model, dataset_all,
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
