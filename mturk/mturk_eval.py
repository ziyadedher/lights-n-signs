from boto.mturk.connection import MTurkConnection
import json
import cv2
import urllib

MIN_BOX_SIZE = 10 # Filter out misclicks

# Create your connection to MTurk
mtc = MTurkConnection(aws_access_key_id='AKIAJUXDOI7EZEMQMDEQ',
aws_secret_access_key='hw8BWplLlTdhyFsbCuU6HYy1zAAfgznCP9Xuz8x0',
host='mechanicalturk.amazonaws.com')

# Annotations come in form: [{hit_id: <>, image_id: <>, image_url: <>, label: <>, bounding_box: {left:<>, top:<>, width:<>, height:<>}}, ...]

with open("annotations/mturk_10mph.json", "r") as f:
    tasks = json.load(f)

completed = 0
incomplete = 0
workers = set()

for task in tasks:
    result = mtc.get_assignments(task["hit_id"])
    if len(result) == 0:
        incomplete += 1
        print("No workers did HIT {} yet".format(task["hit_id"]))
        continue
    completed += 1
    assignment = result[0]
    worker_id = assignment.WorkerId
    workers.add(worker_id)
    for answer in assignment.answers[0]:
      if answer.qid == 'annotation_data':
        worker_answer = json.loads(answer.fields[0])

    print("The Worker with ID {} gave the answer {}".format(worker_id, worker_answer))

    task["bounding_box"] = []
    for bounding_box in worker_answer:
        if bounding_box['width'] * bounding_box['height'] > MIN_BOX_SIZE:
            task["bounding_box"].append(bounding_box)

    if len(task["bounding_box"]) > 0:
        box = task["bounding_box"][0]
        # urllib.urlretrieve(task["image_url"], "images/" + task["image_id"] + ".jpg")
        # image = cv2.imread("images/" + task["image_id"] + ".jpg", cv2.IMREAD_COLOR)
        # width = image.shape[1]
        # cv2.rectangle(image,(width - box['top'],box['left']),(width - (box['top'] + box['height']), box['left'] + box['width']),(0,255,0),5)
        # cv2.namedWindow(task["image_id"], cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(task["image_id"], 600,600)
        # cv2.imshow(task["image_id"], image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

print("{} out of {} tasks completed by {} different workers".format(completed, completed + incomplete, len(workers)))

with open("annotations/mturk_10mph.json", "w+") as f:
    json.dump(tasks, f)
