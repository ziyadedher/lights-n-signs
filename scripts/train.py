# from lns.common.visualization import visualize
import sys
from lns.common.preprocess import Preprocessor
from lns.yolo.train import YoloTrainer
from lns.yolo.settings import YoloSettings
from lns.yolo.settings import LearningRateType
from lns.yolo.settings import Optimizer

#which datasets to include
include_NUScenes = True
include_JAAD = True
include_SCALE = True

if include_NUScenes:
    dataset = Preprocessor.preprocess ('nuscenes')

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

    dataset = dataset.merge_classes({"pedestrian": dataset.classes})

if include_SCALE:
    if include_NUScenes: #data set already instantiated
        dataset = dataset + Preprocessor.preprocess("SCALE")
        
    else:
        dataset = Preprocessor.preprocess ('SCALE')

    dataset = dataset.merge_classes({"pedestrian": ["ped", "Pedestrian"]})

if include_JAAD:
    if include_NUScenes or include_SCALE: #dataset already been instantiated
        dataset = dataset + Preprocessor.preprocess ('JAADDataset')
    else:
        dataset = Preprocessor.preprocess ('JAADDataset')

    dataset = dataset.merge_classes({'pedestrian':dataset.classes})





print(len(dataset))
print(dataset.classes)


for img, labels in dataset.annotations.items():
    labels = list(filter(lambda label: label.bounds.width > 0 and label.bounds.height > 0, labels))
    dataset.annotations[img] = labels

settings = YoloSettings(
    img_size=(800,640),
    batch_size=16,

    num_threads=2,
    prefetech_buffer=256,

    val_split=0.10,
    val_evaluation_epoch=1,

    warm_up_epoch=3,
    save_epoch=2,

    nms_topk=8,
    nms_threshold=0.5,
    score_threshold=0.25,
    eval_threshold=0.5,

    learning_rate_init=5e-5,
    lr_type=LearningRateType.COSINE_DECAY,
    optimizer_name=Optimizer.ADAM,

    # restore_exclude=None,
    # restore_include=[],
    # update_part=None,
)

trainer = YoloTrainer("yolo_ped_mbd_trial_17", dataset, load=True)
trainer.train(settings)


#model = trainer.generate_model()

#visualize(trainer.dataset, model, show_truth=False)

# for image in trainer.dataset.images:
#     print(image)
#     for prediction in model.predict_path(image):
#         b = prediction.bounds
#         print(b.left, b.top, b.width, b.height, prediction.class_index, prediction.score)
