from typing import List
from tqdm import tqdm

from lns.common.model import Model, PredictedObject2D, Bounds2D
from lns.squeezedet.model import SqueezeDetModel
# from lns.common.preprocessing.lisa import preprocess

import cv2 as cv
import numpy as np

import time

# Tunable parameters
PADDING = 2
CANNY_START_LOW = 5
CANNY_INCREMENT = 3
CANNY_RATIO = 3
RED_GREEN_THRESHOLD = 1.0
CUTOFF_THRESHOLD = 200
FOCUS_FACTOR = 0.2 #between 0 and 0.5
MIN_CONTOUR_DIST = 0.2 # At least 20% of image height

class LightStateModel(Model):

    def __init__(self, checkpoint_path: str) -> None:
        """Classifier to label light color after applying SqueezeDet model"""

        self.squeezedet_model = SqueezeDetModel(checkpoint_path)

    def predict(self, image: np.ndarray) -> List[PredictedObject2D]:
        """Predict the color of the on light"""

        output = []
        preds = self.squeezedet_model.predict(image)
        for prediction in preds:
            # Get on light's bounding box
            box = prediction.bounding_box
            light = image[int(box.top):int(box.bottom), int(box.left):int(box.right)]
            rect = self.find_light(light)[0]

            # Label color and save prediction
            if rect == 'off': color = 'off'
            else: color = self.get_color(light, rect)[0]
            output.append(PredictedObject2D(box, [color]))
        return output

    @classmethod
    def find_light(self, img: np.ndarray, num_lights: int = 1):
        # Work on LAB colorspace channel
        img = cv.GaussianBlur(img, (5, 5), 0)
        lightness = cv.cvtColor(img, cv.COLOR_BGR2LAB)[:, :, 1]

        # Loop till the salient contour is found
        low, high, num_cnts = CANNY_START_LOW, CANNY_START_LOW * CANNY_RATIO, -1
        backup = None
        while num_cnts > num_lights or num_cnts == -1:
            # Detect edges
            canny = cv.Canny(lightness, low, high)
            # Find contours and sort them
            cnts = cv.findContours(canny, method=cv.CHAIN_APPROX_SIMPLE, mode=cv.RETR_LIST)[1]
            cnts = sorted(cnts, key=cv.contourArea, reverse=True)
            # Prune very close contours in the top 5
            if num_lights > 1 and len(cnts) >= num_lights:
                i = 1
                rect = cv.boundingRect(cnts[0])
                while i < min(len(cnts), 5):
                    other = cv.boundingRect(cnts[i])
                    origin_distance = abs(other[1]-rect[1]) + abs(other[0]-rect[0])
                    if origin_distance < MIN_CONTOUR_DIST * lightness.shape[0]:
                        cnts.pop(i) # Contours too close, prune it
                    i += 1
            # Break if we ran out of contours
            if len(cnts) < num_lights:
                if num_cnts == -1: return ['off']
                else: break
            # Tighten edge detection to improvise
            num_cnts = len(cnts)
            backup = cnts
            low += CANNY_INCREMENT
            high = low * CANNY_RATIO

        # Define bounding boxes
        rects = []
        for cnt in backup[:num_lights]:
            x, y, w, h = cv.boundingRect(cnt)
            rects.append((max(0, x-PADDING), max(0, y-PADDING), w+PADDING*2, h+PADDING*2))

        # Return the best contour(s) found
        return rects

    @classmethod
    def get_color(self, img, rect):
        # Expand the region of interest
        x, y, w, h = rect
        square = cv.resize(img[y:y+h,x:x+w,:], (200, 200))

        # Calculate mean color
        b, g, r = square[:, :, 0], square[:, :, 1], square[:, :, 2]
        w, h = b.shape
        
        b, g, r = b[int(w*FOCUS_FACTOR):int((1-FOCUS_FACTOR)*w), int(h*FOCUS_FACTOR):int((1-FOCUS_FACTOR)*h)], \
                  g[int(w*FOCUS_FACTOR):int((1-FOCUS_FACTOR)*w), int(h*FOCUS_FACTOR):int((1-FOCUS_FACTOR)*h)], \
                  r[int(w*FOCUS_FACTOR):int((1-FOCUS_FACTOR)*w), int(h*FOCUS_FACTOR):int((1-FOCUS_FACTOR)*h)]

        tb, tg, tr = b > CUTOFF_THRESHOLD, g > CUTOFF_THRESHOLD, r > CUTOFF_THRESHOLD
        b, g, r = (b * tb + b*0.01).mean(), (g*tg + g*0.01).mean(), (r*tr + r*0.01).mean()

        # Make a decision
        if (r) / (g) < RED_GREEN_THRESHOLD: return 'green', square
        else: return 'red', square

def test_classification(dataset: str):
    from lns.common.preprocess import Preprocessor
    dataset = Preprocessor.preprocess(dataset)
    print(dataset.classes)
    dataset = dataset.merge_classes({
        'green': ['Green'],
        'red': ['Red', 'Yellow']
    })
    success = 0
    count = 0
    green_for_red = 0
    red_for_green = 0
    other = 0
    average_width = 0
    average_height = 0
    average_wrong_height = 0
    average_wrong_width = 0

    window = cv.namedWindow('image')
    print(dataset.classes)
    for path, annotation in tqdm(dataset.annotations.items()):
        img = cv.imread(path)
        for a in annotation:
            x1 = a['x_min']
            x2 = a['x_max']
            y1 = a['y_min']
            y2 = a['y_max']
            h = (y2 - y1) * 0
            w = (x2 - x1) * 0
            h_max = img.shape[0]
            w_max = img.shape[1]
            x1 = max(0,int(x1 - w/2))
            x2 = min(w_max,int(x2 + w/2))
            y1 = max(0,int(y1 - h/2))
            y2 = min(h_max,int(y2 + h/2))
            partial = img[y1:y2, x1:x2, :]
            cv.imshow('image', partial)
            #print(dataset.classes[a['class']])
            rect = LightStateModel.find_light(partial)[0]
            if rect == 'off': 
                color = 'off'
            else: color = LightStateModel.get_color(partial, rect)[0]

            if color == dataset.classes[a['class']]:
                success += 1
                average_height += partial.shape[1]
                average_width += partial.shape[0]
            else:
                #print("{} confused for {} for image of size {}".format(
                #    dataset.classes[a['class']], color, partial.shape
                #))
                #cv.imshow('image', partial)
                #cv.waitKey()

                average_wrong_width += partial.shape[1]
                average_wrong_height += partial.shape[0]
                if dataset.classes[a['class']] == 'green' and color == 'red':
                    green_for_red += 1
                elif dataset.classes[a['class']] == 'red' and color == 'green':
                    red_for_green += 1
                else:
                    other += 1

            count += 1
            #print(success/count)

    average_wrong_width /= (count - success)
    average_wrong_height /= (count - success)
    average_width /= success
    average_height /= success

    print("{} of the images were green lights confused for red lights".format(green_for_red/count))
    print("{} of the images were red lights confused for green lights".format(red_for_green/count))
    print("{} of the images were other mistakes".format(other/count))
    print("The wrong images had an average pixel size of {},{}".format(average_wrong_width, average_wrong_height))
    print("The correct images had an average pixel size of {},{}".format(average_width, average_height))

    return success / count

            #cv.waitKey(0)

def benchmark(model, dataset, *, proportion: float = 0.1, overlap_threshold: float = 0.5):
    # Get the classes and annotations and generate the confusion matrix with the `none` class
    classes = dataset.classes + ["__none__"]
    annotations = dataset.annotations
    confusion_matrix = {
        class_name: {
            class_name: 0
            for class_name in classes
        }
        for class_name in classes
    }

    total_iou = 0.0
    count = 0

    # Flatten the images from the different datasets and iterate through each
    image_paths = [
        image_path for image_paths in dataset.image_split(proportion)[0].values()
        for image_path in image_paths
    ]
    class_error = 0
    class_count = 0
    total_processed = 0
    for image_path in tqdm(image_paths):
        total_processed += 1
        # Grab the ground truths for this image and package them under a `PredictedObject2D`
        # to make statistics easier to work with
        ground_truths = [PredictedObject2D(
            Bounds2D(
                label["x_min"], label["y_min"],
                label["x_max"] - label["x_min"], label["y_max"] - label["y_min"]
            ),
            [classes[label["class"]]]
        ) for label in annotations[image_path]]
        # Keep track of which ground truths were found to get false negatives
        detected = [False] * len(ground_truths)

        # Predict on this image and iterate through each prediction to check for matches
        image = cv.imread(image_path)
        predictions = model.predict(image)
        for prediction in predictions:
            if 0.15*image.shape[1] > prediction.bounding_box.left or 0.85*image.shape[1] < (prediction.bounding_box.left + prediction.bounding_box.width):
                continue
            any_detected = False

            # Look through
            for i, ground_truth in enumerate(ground_truths):
                iou = prediction.bounding_box.iou(ground_truth.bounding_box)
                overlapping = iou >= overlap_threshold
                same_class = prediction.predicted_classes[0] == ground_truth.predicted_classes[0]

                if overlapping:
                    any_detected = True
                    if not detected[i]:
                        confusion_matrix[ground_truth.predicted_classes[0]][prediction.predicted_classes[0]] += 1
                        detected[i] = True
                    if same_class:
                        total_iou += iou
                        count += 1
                    else:
                        class_error += 1
                    class_count += 1
                    

            if not any_detected:
                confusion_matrix["__none__"][prediction.predicted_classes[0]] += 1

        for i, is_detected in enumerate(detected):
            if not is_detected:
                confusion_matrix[ground_truths[i].predicted_classes[0]]["__none__"] += 1

    return total_iou / count, confusion_matrix


def print_confusion_matrix(confusion_matrix, spaces: int = 12) -> None:
    names = list(confusion_matrix.keys())

    print("\n\n")
    print("true\\pred".center(spaces), end="")
    for column in names:
        print(f"{column}".ljust(spaces), end="")
    print("")
    for name, nums in confusion_matrix.items():
        print(f"{name}".ljust(spaces), end="")
        for num in nums.values():
            print(f"{num}".ljust(spaces), end="")
        print("")

    print("\n\n")
    stats: Dict[str, Dict[str, float]] = {}
    aggregate_true_positive = 0
    aggregate_false_positive = 0
    aggregate_false_negative = 0
    for name in names:
        stats[name] = {}

        true_positive = confusion_matrix[name][name]
        false_positive = sum(confusion_matrix[other_name][name] for other_name in names if other_name != name)
        false_negative = sum(confusion_matrix[name][other_name] for other_name in names if other_name != name)
        if name != '__none__':
            aggregate_true_positive += true_positive
            aggregate_false_positive += false_positive
            aggregate_false_negative += false_negative

        stats[name]["precision"] = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_negative) != 0 else 0
        )
        stats[name]["recall"] = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) != 0 else 0
        )
        stats[name]["f1"] = 2 * (
            (stats[name]["precision"] * stats[name]["recall"]) / (stats[name]["precision"] + stats[name]["recall"])
        ) if stats[name]["precision"] + stats[name]["recall"] != 0 else 0

    for stat_name in ("class", "precision", "recall", "f1"):
        print(f"{stat_name}".ljust(spaces), end="")
    print("")
    for name, stat in stats.items():
        print(f"{name}".ljust(spaces), end="")
        for value in stat.values():
            print(f"{value:.5f}".ljust(spaces), end="")
        print("")

    print("Total precision: " + str(aggregate_true_positive/(aggregate_true_positive + aggregate_false_positive)))
    print("Total recall: " + str(aggregate_true_positive/(aggregate_true_positive + aggregate_false_negative)))


# if __name__ == '__main__':
#     CHECKPOINT_PATH = '/home/lns/lns/xiyan/models/alllights-414000/train/model.ckpt-415500'

#     from lns.common.preprocess import Preprocessor
#     dataset = Preprocessor.preprocess("scale_lights")
#     dataset = dataset.merge_classes({
#         'green': ['Green'],
#         'red': ['Red', 'Yellow']
#     })
#     print(dataset.classes)
    
#     model = LightStateModel(CHECKPOINT_PATH)
#     average_iou, confusion_matrix = benchmark(model, dataset, proportion=0.1, overlap_threshold=0.1)
#     print_confusion_matrix(confusion_matrix)
#     print(f"\naverage IOU: {average_iou:.6f}")

if __name__ == "__main__":
    print(test_classification('scale_lights'))
