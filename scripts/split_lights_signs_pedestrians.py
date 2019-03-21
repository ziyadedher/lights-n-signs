import sys
import pickle
import os
all_scale_pickle_path = sys.argv[1]

OBJECTS_CLASSES = ['Car/Pickup Truck', 'Pedestrian', 'Other Vehicle', 'Cyclist']
LIGHTS_CLASSES = ['Red', 'Green', 'Yellow']
SIGNS_CLASSES = ['Handicap Parking', 'Stop']

LIGHTS_DATASET = {}
LIGHTS_ANNOTATIONS = {}
LIGHTS_IMAGES = []

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

LIGHTS_DATASET['images'] = LIGHTS_IMAGES
LIGHTS_DATASET['classes'] = LIGHTS_CLASSES
LIGHTS_DATASET['annotations'] = LIGHTS_ANNOTATIONS

print(len(LIGHTS_IMAGES))
with open(os.path.join(os.path.dirname(all_scale_pickle_path), 'scale_lights.pickle'), "wb") as handle:
    pickle.dump(LIGHTS_DATASET, handle, protocol=pickle.HIGHEST_PROTOCOL)

