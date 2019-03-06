import scaleapi
import pickle
from tqdm import tqdm


client = scaleapi.ScaleClient('live_c51c4273f60f4bcb9e86578c372aa51d')

"""
{
      "task_id": "576ba301eed30241b0e9bbf7",
      "created_at": "2016-06-23T08:51:13.903Z",
      "callback_url": "http://www.example.com/callback",
      "type": "categorization",
      "status": "completed",
      "instruction": "Is this object big or small?",
      "urgency": "standard",
      "params": {
        "attachment_type": "text",
        "attachment": "T-Rex",
        "categories": [
          "big",
          "small"
        ]
      }
}

{
    "images": [
        "/absolute/path/to/image.png",
        "/absolute/path/to/other_image.png",
        "/absolute/path/to/nested/image.png",
        ...
    ],

    "classes": [
        "class_one",
        "other_detection_class",
        ...
    }

    "annotations": {
        "/absolute/path/to/image.png": [
            {
                "class": index_of_class,
                "x_min": x_coordinate_of_top_left_corner,
                "y_min": y_coordinate_of_top_left_corner,
                "x_max": x_coordinate_of_bottom_right_corner,
                "y_max": y_coordinate_of_bottom_right_corner
            },
            ...
        ],
        ...
    }
}
"""
DATASET = {}
ANNOTATIONS = {}
CLASSES = []
IMAGES = []

offset = 0
have_next_page = True
while have_next_page:
    tasklist = client.tasks(status="completed", offset=offset)
    print(len(tasklist))

    for obj in tqdm(tasklist):
        task_id = obj.param_dict['task_id']
        task = client.fetch_task(task_id)
        bbox_list = task.param_dict['response']['annotations']
        img_url = task.param_dict['params']['attachment']

        ANNOTATIONS[img_url] = []
        # ignore empty images
        if len(bbox_list) != 0:
            for bbox in bbox_list:
                box_dict = {}
                object_class = bbox['label']
                if object_class not in CLASSES:
                    CLASSES.append(str(object_class))
                box_dict['class'] = CLASSES.index(object_class)
                box_dict['x_min'] = int(bbox['left'])
                box_dict['y_min'] = int(bbox['top'])
                box_dict['x_max'] = int(bbox['left']) + int(bbox['width'])
                box_dict['y_max'] = int(bbox['top']) + int(bbox['height'])

                ANNOTATIONS[img_url].append(box_dict)

        IMAGES.append(img_url)
        # print("Processed {}".format(img_url))

    if len(tasklist) < 100:
        have_next_page = False

DATASET['images'] = IMAGES
DATASET['classes'] = CLASSES
DATASET['annotations'] = ANNOTATIONS

with open('scale_dataset.pickle', 'wb') as handle:
    pickle.dump(DATASET, handle, protocol=pickle.HIGHEST_PROTOCOL)
