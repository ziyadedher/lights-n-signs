from lns_common.preprocess.preprocess import Preprocessor
from lns_squeezedet.train import SqueezeDetTrainer

mturk_dataset = Preprocessor.preprocess("mturk", scale=0.5)

print(mturk_dataset.classes)

mmap = {"sign": ['Left arrow', 'SPEED LIMIT 20', 'RIGHT TURN ONLY', 'DO NOT ENTER', 'SPEED LIMIT 15', 'SPEED LIMIT 10', 'Handicapped parking', 'SPEED LIMIT 5', 'Right arrow', 'LEFT TURN ONLY']}

mturk = mturk_dataset.merge_classes(mmap)

name = "generic-signs"
trainer = SqueezeDetTrainer(name, mturk)
trainer.setup_squeezedet()
