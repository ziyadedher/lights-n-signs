import cv2
from lns.yolo.train import YoloTrainer
from scripts.visualize import put_labels_on_image

# Get the tensorflow model from trainer
# For dla, this would read the trainer we put under ./lns-training/resource/traineryolo/
trainer = YoloTrainer("new_dataset_ac_21_dla")

# Check the checkpoint path and get the model
print("Model path:" + trainer.get_weights_path())
model = trainer.model

# Prediction parameter setup setup
color_mapping = {
    # pred_class: any("red", "green", "yellow", "off") # This for coloring only
    k: v for k, v in zip(['5-red-green', '4-red-green', 'red', '5-red-yellow', 'green', 'yellow', 'off'],
                         ['red'] * 4 + ['green'] + ['yellow'] + ['off'])
}
threshold = 0.2

# Read Image from repo root
image_path = "../image.png"
image = cv2.imread(image_path)
image = image[0:1377, 0:2448]
# Show the sample image
# cv2.imshow("visualization", image)
# cv2.waitKey()

# Run prediction and put labels on image
# The model.predict(image) is the step to run tensorflow model


model.predict_featuremap(image)