import cv2
from lns.yolo.train import YoloTrainer
from scripts.visualize import put_labels_on_image

trainer = YoloTrainer("new_dataset_ac_21")
model = trainer.model
print(trainer.get_weights_path())


color_mapping = {
    # pred_class: any("red", "green", "yellow", "off") # This for coloring only
    k: v for k, v in zip(['5-red-green', '4-red-green', 'red', '5-red-yellow', 'green', 'yellow', 'off'],
                         ['red'] * 4 + ['green'] + ['yellow'] + ['off'])
}
image_path = "../image.png"
threshold = 0.2
image = cv2.imread(image_path)
# cv2.imshow("visualization", image)
# cv2.waitKey()
image2 = put_labels_on_image(image, model.predict(image), trainer.dataset.classes, is_pred=True, color_mapping=color_mapping, threshold=threshold)
cv2.imshow("visualization", image2)
cv2.waitKey()