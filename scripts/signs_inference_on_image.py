import sys 
import cv2
import os
import time
from pathlib import Path
import random

def augment_brightness(img, max_brightness_change=185):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # generate random brightness noise
    value = max_brightness_change # random.randint(1, max_brightness_change + 1)

    # Clap the result to 0 - 255
    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        value = int(-value)
        lim = 0 + value
        v[v < lim] = 0
        v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def inference_on_image(input_image_path, output_image_path, model_path, num_neighbors=3, scale=1.1, change_brightness=False, equalize_hist=False, vis_weight=True):
    # Determine name of folder where to save results, create directory if needed
    if not os.path.exists(os.path.dirname(output_image_path)):
        os.makedirs(os.path.dirname(output_image_path))

    gray_img = cv2.imread(str(input_image_path), 0)
    out_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    print(out_img[-1])
    if change_brightness:
        out_img = augment_brightness(out_img, max_brightness_change=change_brightness)
        gray_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2GRAY)
    if equalize_hist:
        gray_img = cv2.equalizeHist(gray_img)
        out_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    # im_name = input_image_path[str(input_image_path).rindex('/') + 1:]
    
    # Get the model's detections
    cascade = cv2.CascadeClassifier(model_path)  # Load model
    detections, rejectlevels, weights = cascade.detectMultiScale3(gray_img, scale, num_neighbors, outputRejectLevels=True)

    # Draw bounding boxes
    for (x_det, y_det, w_det, h_det) in detections:
        # Pred bounding box -> cyan
        cv2.rectangle(out_img, (x_det, y_det), (x_det+w_det, y_det+h_det), (255, 255, 0), 2)
        if vis_weight:
            out_img = cv2.putText(out_img, # Put label on the image
                                  f"{weights[0][0]}",
                                  (x_det, y_det), cv2.FONT_HERSHEY_PLAIN,
                                  1, (255, 255, 0), thickness=2)
    print(detections, rejectlevels, weights)
    cv2.imwrite(output_image_path, out_img)


if __name__=='__main__':
    input_image_path = "/home/lns/tiffany/lights-n-signs-training/visualization/17.png"
    output_image_path = "/home/lns/tiffany/lights-n-signs-training/visualization/17_inference.png"
    model_path = "/home/lns/.lns-training/resources/trainers/haar/sam-3-haar-feb-23-05/cascade/cascade.xml"
    min_neighbors = 2
    scale = 1.08
    inference_on_image(input_image_path, output_image_path, model_path, num_neighbors=min_neighbors, scale=scale)
    input_image_path = "/home/lns/tiffany/lights-n-signs-training/visualization/17.png"
    output_image_path = "/home/lns/tiffany/lights-n-signs-training/visualization/17_inference_eqhist.png"
    model_path = "/home/lns/.lns-training/resources/trainers/haar/sam-3-haar-feb-23-05/cascade/cascade.xml"
    min_neighbors = 2
    scale = 1.08
    inference_on_image(input_image_path, output_image_path, model_path, num_neighbors=min_neighbors, scale=scale, equalize_hist=True)

    output_image_path = "/home/lns/tiffany/lights-n-signs-training/visualization/17_inference_darker.png"
    model_path = "/home/lns/.lns-training/resources/trainers/haar/sam-3-haar-feb-23-05/cascade/cascade.xml"
    min_neighbors = 2
    scale = 1.08
    inference_on_image(input_image_path, output_image_path, model_path, num_neighbors=min_neighbors, scale=scale, change_brightness=-185)

    output_image_path = "/home/lns/tiffany/lights-n-signs-training/visualization/17_inference_brighter.png"
    model_path = "/home/lns/.lns-training/resources/trainers/haar/sam-3-haar-feb-23-05/cascade/cascade.xml"
    min_neighbors = 2
    scale = 1.08
    inference_on_image(input_image_path, output_image_path, model_path, num_neighbors=min_neighbors, scale=scale, change_brightness=70)

    output_image_path = "/home/lns/tiffany/lights-n-signs-training/visualization/17_inference_darker_eqhist.png"
    model_path = "/home/lns/.lns-training/resources/trainers/haar/sam-3-haar-feb-23-05/cascade/cascade.xml"
    min_neighbors = 2
    scale = 1.08
    inference_on_image(input_image_path, output_image_path, model_path, num_neighbors=min_neighbors, scale=scale, change_brightness=-185, equalize_hist=True)

    output_image_path = "/home/lns/tiffany/lights-n-signs-training/visualization/17_inference_brighter_eqhist.png"
    model_path = "/home/lns/.lns-training/resources/trainers/haar/sam-3-haar-feb-23-05/cascade/cascade.xml"
    min_neighbors = 2
    scale = 1.08
    inference_on_image(input_image_path, output_image_path, model_path, num_neighbors=min_neighbors, scale=scale, change_brightness=70, equalize_hist=True)