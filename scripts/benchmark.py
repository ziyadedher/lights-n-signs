from typing import Tuple, Dict

import cv2  # type: ignore
from tqdm import tqdm  # type: ignore

from lns.common.model import Model, Bounds2D, PredictedObject2D
from lns.common.dataset import Dataset
from lns.common.utils.visualization import put_labels_on_image, put_predictions_on_image


ConfusionMatrix = Dict[str, Dict[str, int]]

cv2.namedWindow("visualization", cv2.WINDOW_NORMAL)
cv2.resizeWindow("visualization", 1920, 1080)


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
    for image_path in tqdm(image_paths):
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
        image = cv2.imread(image_path)
        predictions = model.predict(image)
        for prediction in predictions:
            any_detected = False

            # Look through
            for i, ground_truth in enumerate(ground_truths):
                iou = prediction.bounding_box.iou(ground_truth.bounding_box)
                overlapping = iou >= overlap_threshold
                same_class = prediction.predicted_classes[0] == ground_truth.predicted_classes[0]

                if same_class and overlapping:
                    if not detected[i]:
                        confusion_matrix[ground_truth.predicted_classes[0]][prediction.predicted_classes[0]] += 1
                        detected[i] = True
                    any_detected = True
                    total_iou += iou
                    count += 1

            if not any_detected:
                image = put_labels_on_image(image, annotations[image_path])
                image = put_predictions_on_image(image, [prediction])
                cv2.imshow("visualization", image)
                key = cv2.waitKey(0)
                while key != 10:
                    key = cv2.waitKey(0)
                confusion_matrix["__none__"][prediction.predicted_classes[0]] += 1

        for i, is_detected in enumerate(detected):
            if not is_detected:
                confusion_matrix[ground_truths[i].predicted_classes[0]]["__none__"] += 1

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
    for name in names:
        stats[name] = {}

        true_positive = confusion_matrix[name][name]
        false_positive = sum(confusion_matrix[other_name][name] for other_name in names if other_name != name)
        false_negative = sum(confusion_matrix[name][other_name] for other_name in names if other_name != name)

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


if __name__ == '__main__':
    from lns.common.preprocess import Preprocessor
    bosch = Preprocessor.preprocess("Bosch")
    lights = Preprocessor.preprocess("lights")
    dataset = lights
    dataset = dataset.merge_classes({
        "green": [
            "GreenLeft", "Green", "GreenRight", "GreenStraight",
            "GreenStraightRight", "GreenStraightLeft", "Green traffic light"
        ],
        "red": [
            "Yellow", "RedLeft", "Red", "RedRight", "RedStraight",
            "RedStraightLeft", "Red traffic light", "Yellow traffic light"
        ],
        "off": ["off"]
    })
    dataset = dataset.minimum_area(0.0001)

    from lns.squeezedet.model import SqueezeDetModel
    model = SqueezeDetModel("/home/lns/lns/xiyan/models/alllights-414000/train/model.ckpt-415500")

    average_iou, confusion_matrix = benchmark(model, dataset, proportion=0.1, overlap_threshold=0.1)
    print_confusion_matrix(confusion_matrix)
    print(f"\naverage IOU: {average_iou:.6f}")
