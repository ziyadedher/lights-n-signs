# from lns.common.preprocess import Preprocessor
# dataset = Preprocessor.preprocess('ScaleLights')

# from lns.yolo.train import YoloTrainer
# trainer = YoloTrainer('darknet25_fullres_tiffany',dataset) #,load=True)

# trainer.train()

#### train squeezedet full res
# from lns.common.preprocess import Preprocessor
# dataset_scale = Preprocessor.preprocess('ScaleLights')
# dataset_utias = Preprocessor.preprocess('ScaleLights_New_Utias')
# dataset_youtube = Preprocessor.preprocess('ScaleLights_New_Youtube')

# dataset_scale_utias = dataset_scale.__add__(dataset_utias)
# dataset_all = dataset_scale_utias.__add__(dataset_youtube)
# dataset_all = dataset_all.merge_classes({
#     "green": ["goLeft", "Green", "GreenLeft", "GreenStraightRight", "go", "GreenStraightLeft", "GreenRight", "GreenStraight", "3-green", "4-green", "5-green"],
#     "yellow": ["warning", "Yellow", "warningLeft", "3-yellow", "4-yellow", "5-yellow"],
#     "red": ["stop", "stopLeft", "RedStraightLeft", "Red", "RedLeft", "RedStraight", "RedRight", "3-red", "4-red", "5-red"],
#     "off": ["OFF", "off", "3-off", "3-other", "4-off", "4-other", "5-off", "5-other"]
#     })


# from lns.squeezedet.train import SqueezedetTrainer
# trainer = SqueezedetTrainer('squeezedet_fullres_tiffany',dataset_all)

# trainer.train()

#### train haar on lisa and scale
from lns.common.preprocess import Preprocessor
from lns.haar.train import HaarTrainer

dataset_lisa = Preprocessor.preprocess('lisa_signs')
dataset_scale = Preprocessor.preprocess('ScaleSigns', force=True)
print(dataset_scale)
dataset = dataset_lisa.__add__(dataset_scale)
# dataset = dataset.merge_classes({

# })
print(dataset.classes)
exit()
trainer = HaarTrainer('tiffany_test_all',dataset)
trainer.setup()
trainer.train()

"""
===== TRAINING 29-stage =====
<BEGIN
POS count : consumed   1000 : 1149
NEG count : acceptanceRatio    500 : 3.98295e-08
Precalculation time: 0
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|    0.998|    0.976|
+----+---------+---------+
|   5|    0.998|    0.976|
+----+---------+---------+
|   6|        1|     0.98|
+----+---------+---------+
|   7|        1|    0.984|
+----+---------+---------+
|   8|        1|    0.976|
+----+---------+---------+
|   9|    0.998|    0.976|
+----+---------+---------+
|  10|    0.999|     0.98|
+----+---------+---------+
|  11|    0.999|    0.952|
+----+---------+---------+
|  12|    0.999|     0.96|
+----+---------+---------+
|  13|    0.996|    0.928|
+----+---------+---------+
|  14|    0.996|    0.946|
+----+---------+---------+
|  15|    0.998|    0.972|
+----+---------+---------+
|  16|    0.996|    0.936|
+----+---------+---------+
|  17|    0.996|    0.904|
+----+---------+---------+
|  18|    0.996|    0.884|
+----+---------+---------+
|  19|    0.996|     0.86|
+----+---------+---------+
|  20|    0.996|    0.858|
+----+---------+---------+
|  21|    0.996|    0.862|
+----+---------+---------+
|  22|    0.996|    0.916|
+----+---------+---------+
|  23|    0.996|    0.862|
+----+---------+---------+
|  24|    0.996|    0.888|
+----+---------+---------+
|  25|    0.996|    0.848|
+----+---------+---------+
|  26|    0.996|    0.896|
+----+---------+---------+
|  27|    0.996|     0.89|
+----+---------+---------+
|  28|    0.996|    0.866|
+----+---------+---------+
|  29|    0.996|    0.832|
+----+---------+---------+
|  30|    0.996|    0.904|
+----+---------+---------+
|  31|    0.996|     0.88|
+----+---------+---------+
|  32|    0.996|    0.818|
+----+---------+---------+
|  33|    0.996|     0.78|
+----+---------+---------+
|  34|    0.996|    0.774|
+----+---------+---------+
|  35|    0.996|    0.794|
+----+---------+---------+
|  36|    0.996|    0.818|
+----+---------+---------+
|  37|    0.996|    0.752|
+----+---------+---------+
|  38|    0.996|    0.736|
+----+---------+---------+
|  39|    0.996|    0.764|
+----+---------+---------+
|  40|    0.996|    0.768|
+----+---------+---------+
|  41|    0.996|    0.722|
+----+---------+---------+
|  42|    0.996|    0.726|
+----+---------+---------+
|  43|    0.996|    0.714|
+----+---------+---------+
|  44|    0.996|    0.682|
+----+---------+---------+
|  45|    0.996|    0.636|
+----+---------+---------+
|  46|    0.996|    0.618|
+----+---------+---------+
|  47|    0.996|    0.598|
+----+---------+---------+
|  48|    0.996|    0.608|
+----+---------+---------+
|  49|    0.996|    0.582|
+----+---------+---------+
|  50|    0.996|     0.56|
+----+---------+---------+
|  51|    0.996|     0.54|
+----+---------+---------+
|  52|    0.996|    0.568|
+----+---------+---------+
|  53|    0.996|    0.514|
+----+---------+---------+
|  54|    0.996|    0.476|
+----+---------+---------+
END>
Training until now has taken 1 days 21 hours 14 minutes 18 seconds.
Training completed at stage 29.
"""