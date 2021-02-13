from typing import Tuple, Dict

import cv2  # type: ignore
from tqdm import tqdm  # type: ignore

from lns.common.model import Model
from lns.common.structs import Bounds2D, Object2D
from lns.common.dataset import Dataset
from lns.common.visualization import _put_labels_on_image #, put_predictions_on_image


ConfusionMatrix = Dict[str, Dict[str, int]]
#print("before CV2 windows")
#cv2.namedWindow("visualization", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("visualization", 1920, 1080)
#print('here')

def benchmark(model: Model, dataset: Dataset, *,
              proportion: float = 0.1, overlap_threshold: float = 0.5) -> Tuple[float, ConfusionMatrix]:
    # Get the classes and annotations and generate the confusion matrix with the `none` class
    classes = dataset.classes + ["__none__"]
    annotations = dataset.annotations
    confusion_matrix: ConfusionMatrix = {
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
    total_processed = 0
    for image_path in tqdm(image_paths):
        total_processed += 1
        # Grab the ground truths for this image and package them under a `PredictedObject2D`
        # to make statistics easier to work with
        ground_truths = [Object2D(
            Bounds2D(
                label["x_min"], label["y_min"],
                label["x_max"] - label["x_min"], label["y_max"] - label["y_min"]
            ),
            [classes[label["class"]]]
        ) for label in annotations[image_path]]
        # Keep track of which ground truths were found to get false negatives
        detected = [False] * len(ground_truths)

        # Predict on this image and iterate through each prediction to check for matches
        image = cv2.imread(image_path)
        predictions = model.predict(image)
        image = put_labels_on_image(image, annotations[image_path])
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

            #image = put_predictions_on_image(image, [prediction])
            image = put_labels_on_image(image, [prediction])
            if not any_detected:
                confusion_matrix["__none__"][prediction.predicted_classes[0]] += 1

        for i, is_detected in enumerate(detected):
            if not is_detected:
                confusion_matrix[ground_truths[i].predicted_classes[0]]["__none__"] += 1
        # cv2.imshow("visualization", image)
        # key = cv2.waitKey(0)
        # while key != 27:
        #     key = cv2.waitKey(0)

    return total_iou / count, confusion_matrix


def print_confusion_matrix(confusion_matrix: ConfusionMatrix, spaces: int = 12) -> None:
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


if __name__ == '__main__':
    from lns.common.preprocess import Preprocessor
    print('processing Dataset')
    dataset = Preprocessor.preprocess('SCALE')

    dataset = dataset.merge_classes({"pedestrian": ["ped", "Pedestrian"]})
    
    # dataset = dataset.minimum_area(0.0005)
    # dataset = dataset.remove_perpendicular_lights(0.7)

    #from lns.common.cv_lights import LightStateModel
    print('importing model')
    from lns.yolo.model import YoloModel
    model = YoloModel("/home/od/.lns-training/resources/trainers/yolo/yolo_ped_mbd_trial_14/checkpoint/model-epoch_16_step_144244_loss_0.5045_lr_1.2239e-06",'placeholder','placeholder') #.data-00000-of-00001

    average_iou, confusion_matrix = benchmark(model, dataset, proportion=0.1, overlap_threshold=0.1)
    print_confusion_matrix(confusion_matrix)
    print(f"\naverage IOU: {average_iou:.6f}")
