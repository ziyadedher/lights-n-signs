import sys
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pickle_path', type=str, help='path to scale Dataset pickle file')
# parser.add_argument('--project', type=str, help='one of [light, sign, all]')

args = parser.parse_args()
all_scale_pickle_path = args.pickle_path
# project = args.project

OBJECTS_CLASSES = ['Car/Pickup Truck', 'Pedestrian', 'Other Vehicle', 'Cyclist', 'Railroad Sign', 'Railroad Light Pair On', 'Railroad Light Pair Off']
LIGHTS_CLASSES = ['Red', 'Green', 'Yellow']
SIGNS_CLASSES = ['Handicap Parking', 'Stop', 'Do Not Enter', 'Left Turn Only (arrow)', 'Left Turn Only (words)', 'Right Turn Only (words)', 'Right Turn Only (arrow)', 'Speed Limit 5', 'Speed Limit 10', 'Speed Limit 15', 'Speed Limit 20']

# ['Speed Limit 10', 'Speed Limit 5', 'Stop', 'Right Turn Only (words)', 'Right Turn Only (arrow)', 'Left Turn Only (words)', 'Left Turn Only (arrow)', 'Handicap Parking', 'Do Not Enter']


LIGHTS_DATASET = {}
LIGHTS_ANNOTATIONS = {}
LIGHTS_IMAGES = []

SIGNS_DATASET = {}
SIGNS_ANNOTATIONS = {}
SIGNS_IMAGES = []

with open(all_scale_pickle_path, "rb") as f:
    data = pickle.load(f)
    print(data['classes'])
    for image in data['images']:
        annotation = data['annotations'][image]
        if len(annotation) == 0: 
            continue        
        one_class_index = annotation[0]['class']
        if data['classes'][one_class_index] in LIGHTS_CLASSES:
            LIGHTS_IMAGES.append(image)
            for bbox in annotation:
                bbox['class'] = LIGHTS_CLASSES.index(data['classes'][bbox['class']])            
            LIGHTS_ANNOTATIONS[image] = annotation
        elif data['classes'][one_class_index] in OBJECTS_CLASSES:
            print("object")
        elif data['classes'][one_class_index] in SIGNS_CLASSES:
            SIGNS_IMAGES.append(image)
            for bbox in annotation:
                bbox['class'] = SIGNS_CLASSES.index(data['classes'][bbox['class']])
            SIGNS_ANNOTATIONS[image] = annotation

LIGHTS_DATASET['images'] = LIGHTS_IMAGES
LIGHTS_DATASET['classes'] = LIGHTS_CLASSES
LIGHTS_DATASET['annotations'] = LIGHTS_ANNOTATIONS

SIGNS_DATASET['images'] = SIGNS_IMAGES
SIGNS_DATASET['classes'] = SIGNS_CLASSES
SIGNS_DATASET['annotations'] = SIGNS_ANNOTATIONS

print(len(LIGHTS_IMAGES))
with open(os.path.join(os.path.dirname(all_scale_pickle_path), 'scale_lights.pickle'), "wb") as handle:
    pickle.dump(LIGHTS_DATASET, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(len(SIGNS_IMAGES))
with open(os.path.join(os.path.dirname(all_scale_pickle_path), 'scale_signs.pickle'), "wb") as handle:
    pickle.dump(SIGNS_DATASET, handle, protocol=pickle.HIGHEST_PROTOCOL)

