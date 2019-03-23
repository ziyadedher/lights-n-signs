from lns_common.preprocess.preprocess import Preprocessor
from lns_squeezedet.train import SqueezeDetTrainer

scale_lights_dataset = Preprocessor.preprocess("scale_lights")

print(scale_lights_dataset.classes)

lmap = {'light': ["Red", "Green", "Yellow"]}

scale_lights = scale_lights_dataset.merge_classes(lmap)

name = "generic-lights"
trainer = SqueezeDetTrainer(name, scale_lights)
trainer.setup_squeezedet()
