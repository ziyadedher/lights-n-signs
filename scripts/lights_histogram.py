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


import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
sns.set_theme()

all_widths = []
all_heights = []
all_list_obj2d = list(dataset_all.annotations.values())

for list_obj2d in tqdm(all_list_obj2d):
    for obj2d in list_obj2d:
        all_widths.append(obj2d.bounds.width)
        all_heights.append(obj2d.bounds.height)

plt.hist(all_widths)
plt.xlabel('Width')
plt.ylabel('Number of images')
plt.tight_layout()
plt.savefig("lights_width_histogram.png")
plt.clf()

plt.hist(all_heights)
plt.xlabel('Height')
plt.ylabel('Number of images')
plt.tight_layout()
plt.savefig("lights_height_histogram.png")

