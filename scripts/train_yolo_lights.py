from lns.common.preprocess import Preprocessor
from lns.yolo.train import YoloTrainer

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

# Get all bbox areas
all_areas = []
all_list_obj2d = list(dataset_all.annotations.values())
for list_obj2d in all_list_obj2d:
    for obj2d in list_obj2d:
        all_areas.append(obj2d.bounds.width * obj2d.bounds.height)

# Get 85th percentile
all_areas.sort()
eighty_fifth = int(0.85 * len(all_areas))
threshold = all_areas[eighty_fifth]

# Remove images with any bbox bigger than 85th percentile
for annotation in list(dataset_all.annotations):
    list_obj2d = dataset_all.annotations[annotation]
    for obj2d in list_obj2d:
        area = obj2d.bounds.width * obj2d.bounds.height
        if area > threshold:
            del dataset_all.annotations[annotation]
            dataset_all.images.remove(annotation)
            break

trainer = YoloTrainer(name="matthieu_darknet53_416_4_copy", dataset=dataset_all, load=True) # Training from scratch
trainer.train()
