# YOLO
# from lns.common.preprocess import Preprocessor
# dataset_scale = Preprocessor.preprocess('ScaleLights')
# dataset_utias = Preprocessor.preprocess('ScaleLights_New_Utias')
# dataset_youtube = Preprocessor.preprocess('ScaleLights_New_Youtube')

# dataset_all = dataset_scale + dataset_utias + dataset_youtube
# dataset_all = dataset_all.merge_classes({
#     "green": ["goLeft", "Green", "GreenLeft", "GreenStraightRight", "go", "GreenStraightLeft", "GreenRight", "GreenStraight", "3-green", "4-green", "5-green"],
#     "yellow": ["warning", "Yellow", "warningLeft", "3-yellow", "4-yellow", "5-yellow"],
#     "red": ["stop", "stopLeft", "RedStraightLeft", "Red", "RedLeft", "RedStraight", "RedRight", "3-red", "4-red", "5-red"],
#     "off": ["OFF", "off", "3-off", "3-other", "4-off", "4-other", "5-off", "5-other"]
#     })


# from lns.yolo.train import YoloTrainer
# trainer = YoloTrainer('darknet25_416_matthieu_trial2', dataset_all)
# trainer.train()

# Squeezedet

from lns.common.preprocess import Preprocessor
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


from lns.squeezedet.train import SqueezedetTrainer
# trainer = SqueezedetTrainer('squeezedet_fullres_tiffany_copy1', dataset_all)
trainer = SqueezedetTrainer('first-real_copy1', dataset_all)

trainer.train()