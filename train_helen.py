from lns.common.preprocess import Preprocessor
dataset_scale = Preprocessor.preprocess('ScaleLights')
dataset_utias = Preprocessor.preprocess('ScaleLights_New_Utias')
dataset_youtube = Preprocessor.preprocess('ScaleLights_New_Youtube')

dataset_scale_utias = dataset_scale.__add__(dataset_utias)
dataset_all = dataset_scale_utias.__add__(dataset_youtube)
dataset_all = dataset_all.merge_classes({
    "green": ["goLeft", "Green", "GreenLeft", "GreenStraightRight", "go", "GreenStraightLeft", "GreenRight", "GreenStraight", "3-green", "4-green", "5-green"],
    "yellow": ["warning", "Yellow", "warningLeft", "3-yellow", "4-yellow", "5-yellow"],
    "red": ["stop", "stopLeft", "RedStraightLeft", "Red", "RedLeft", "RedStraight", "RedRight", "3-red", "4-red", "5-red"],
    "off": ["OFF", "off", "3-off", "3-other", "4-off", "4-other", "5-off", "5-other"]
    })


# from lns.yolo.train import YoloTrainer
# trainer = YoloTrainer('darknet25_640_416_helen',dataset_all)

from lns.squeezedet.train import SqueezedetTrainer
trainer = SqueezedetTrainer('helen_squeezedet_1248_384_1',dataset_all,load=False)

trainer.train()

#### train haar on lisa and scale
#from lns.common.preprocess import Preprocessor
#from lns.haar.train import HaarTrainer

#dataset_lisa = Preprocessor.preprocess('lisa_signs')
#dataset_scale = Preprocessor.preprocess('ScaleSigns', force=True)
# print(dataset_scale)
#dataset = dataset_lisa.__add__(dataset_scale)
# dataset = dataset.merge_classes({

# })
# print(dataset.classes)
# exit()
# trainer = HaarTrainer('helen_test_all',dataset)
# trainer.setup()
# trainer.train()

