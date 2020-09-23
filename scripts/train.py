# from lns.common.visualization import visualize
from lns.common.preprocess import Preprocessor
from lns.yolo.train import YoloTrainer
from lns.yolo.settings import YoloSettings

dataset = Preprocessor.preprocess("nuscenes")
pedestrian_id = dataset.classes.index("pedestrian")

dead_keys = []
for img, labels in dataset.annotations.items():
    labels = list(filter(lambda label: label.class_index == pedestrian_id, labels))
    if not labels:
        dead_keys.append(img)
    else:
        dataset.annotations[img] = labels
for dead_key in dead_keys:
    del dataset.annotations[dead_key]
    dataset.images.remove(dead_key)

dataset = dataset.merge_classes({"ped": dataset.classes})

dataset = dataset + Preprocessor.preprocess("SCALE")
dataset = dataset.merge_classes({"pedestrian": ["ped", "Pedestrian"]})
print(len(dataset))
print(dataset.classes)


# scale = Preprocessor.preprocess("ScaleLights")
# scale_utias = Preprocessor.preprocess("ScaleLights_New_Utias")
# scale_yt = Preprocessor.preprocess("ScaleLights_New_Youtube")
# dataset = scale + scale_utias + scale_yt
# bosch = Preprocessor.preprocess("Bosch")
# lisa = Preprocessor.preprocess("LISA")
# dataset = dataset + bosch + lisa

# print(scale.classes)
# visualize(scale)

# dataset = dataset.merge_classes({
#     "green": ["goLeft", "Green", "GreenLeft", "GreenStraightRight", "go", "GreenStraightLeft", "GreenRight", "GreenStraight", "3-green", "4-green", "5-green"],
#     "yellow": ["warning", "Yellow", "warningLeft", "3-yellow", "4-yellow", "5-yellow"],
#     "red": ["stop", "stopLeft", "RedStraightLeft", "Red", "RedLeft", "RedStraight", "RedRight", "3-red", "4-red", "5-red", "4-red-green", "5-red-green", "5-red-yellow"],
#     "off": ["OFF", "off", "3-off", "3-other", "4-off", "4-other", "5-off", "5-other"]
# })
# dataset = dataset.merge_classes({
#     "green": ["goLeft", "Green", "GreenLeft", "GreenStraightRight", "go", "GreenStraightLeft", "GreenRight", "GreenStraight", "3-green", "4-green", "5-green"],
#     "yellow": ["warning", "Yellow", "warningLeft", "3-yellow", "4-yellow", "5-yellow"],
#     "red": ["stop", "stopLeft", "RedStraightLeft", "Red", "RedLeft", "RedStraight", "RedRight", "3-red", "4-red", "5-red"],
#     "off": ["OFF", "off", "3-off", "3-other", "4-off", "4-other", "5-off", "5-other"]
# })
# dataset = dataset.merge_classes({"light": ["green", "yellow", "red", "off"]})

# print(dataset.classes)
# visualize(dataset)

for img, labels in dataset.annotations.items():
    labels = list(filter(lambda label: label.bounds.width > 0 and label.bounds.height > 0, labels))
    dataset.annotations[img] = labels

settings = YoloSettings(
    img_size=(800, 640),
    batch_size=2,

    num_threads=2,
    prefetech_buffer=256,

    val_split=0.10,
    val_evaluation_epoch=2,

    warm_up_epoch=3,
    save_epoch=2,

    nms_threshold=0.10,
    score_threshold=0.10,
    eval_threshold=0.10,

    restore_exclude=None,
    restore_include=[],
    update_part=None,
)

trainer = YoloTrainer("pedestrian_w_nuscenes_oldyolo", dataset)
trainer.train(settings)


#model = trainer.generate_model()

#visualize(trainer.dataset, model, show_truth=False)

# for image in trainer.dataset.images:
#     print(image)
#     for prediction in model.predict_path(image):
#         b = prediction.bounds
#         print(b.left, b.top, b.width, b.height, prediction.class_index, prediction.score)
