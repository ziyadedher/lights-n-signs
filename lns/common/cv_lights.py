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
            rect = self.find_light(light)

            # Label color and save prediction
            if rect == 'off': color = 'off'
            else: color = self.get_color(light, rect)[0]
            output.append(PredictedObject2D(box, [color]))
        return output

    @classmethod
    def find_light(self, img):
        # Work on LAB colorspace channel
        lightness = cv.cvtColor(img, cv.COLOR_BGR2LAB)[:, :, 1]

        # Loop till the salient contour is found
        low, high, num_cnts = CANNY_START_LOW, CANNY_START_LOW * CANNY_RATIO, -1
        while num_cnts > 1 or num_cnts == -1:
            # Detect edges
            canny = cv.Canny(lightness, low, high)
            # Find contours and sort them
            cnts = cv.findContours(canny, method=cv.CHAIN_APPROX_SIMPLE, mode=cv.RETR_LIST)[1]
            cnts = sorted(cnts, key=cv.contourArea, reverse=True)
            # Break if we ran out of contours
            if len(cnts) == 0:
                if num_cnts == -1: return 'off'
                else: break
            # Tighten edge detection to improvise
            x, y, w, h = cv.boundingRect(cnts[0])
            num_cnts = len(cnts)
            low += CANNY_INCREMENT
            high = low * CANNY_RATIO

        # Return the best contour found
        return max(0, x-PADDING), max(0, y-PADDING), w+PADDING*2, h+PADDING*2

    @classmethod
    def get_color(self, img, rect):
        # Expand the region of interest
        x, y, w, h = rect
        square = cv.resize(img[y:y+h,x:x+w,:], (200, 200))

        # Calculate mean color
        b, g, r = square[:, :, 0], square[:, :, 1], square[:, :, 2]
        #cv.imshow('image', square)
        #cv.waitKey()
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
            rect = LightStateModel.find_light(partial)
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

# if __name__ == '__main__':
#     CHECKPOINT_PATH = '/mnt/ssd2/vinit/model.ckpt-496000'

#     from lns_common.preprocess.preprocess import Preprocessor
#     dataset = Preprocessor.preprocess("LISA")
#     dataset = dataset.merge_classes({
#         'green': ['go', 'goLeft'],
#         'red': ['stop', 'stopLeft', 'warningLeft', 'warning']
#     })
#     print(dataset.classes)
    
#     model = LightStateModel(CHECKPOINT_PATH)
#     benchmark_model(dataset, model)

if __name__ == "__main__":
    print(test_classification('scale_lights'))
