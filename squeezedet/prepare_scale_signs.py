from lns_common.preprocess.preprocess import Preprocessor
from lns_squeezedet.train import SqueezeDetTrainer

dataset = Preprocessor.preprocess("scale_signs")

print(dataset.classes)

# lmap = {'red': ["Red", "Yellow"], 'green': ["Green"]}

# final_dataset = scale_lights_dataset.merge_classes(lmap)

name = "scale-signs-new"
trainer = SqueezeDetTrainer(name, dataset)
trainer.setup_squeezedet()
