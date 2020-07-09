"""nuScenes dataset preprocessor.

The name has an appended _ to avoid a name conflict with the package."""

from itertools import chain

from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor
from lns.common.preprocessing.nuscenes import NuScenes
from lns.common.preprocessing.nuscenes.eval.common.loaders import load_gt
from lns.common.preprocessing.nuscenes.eval.detection.constants import DETECTION_NAMES
from lns.common.preprocessing.nuscenes.eval.detection.data_classes import DetectionBox


DATASET_NAME = "nuScenes"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _nuscenes(path: str) -> Dataset:  # noqa
    images: Dataset.Images = []
    classes: Dataset.Classes = DETECTION_NAMES
    annotations: Dataset.Annotations = {}

    nuscenes = NuScenes(version="v1.0-trainval", dataroot=path)
    gt_boxes_train = load_gt(nuscenes, "train", DetectionBox).all
    gt_boxes_val = load_gt(nuscenes, "val", DetectionBox).all

    for box in chain(gt_boxes_train, gt_boxes_val):
        sd_token = nuscenes.get("sample", box.sample_token)["data"]["CAM_FRONT"]
        img_path = nuscenes.get_sample_data_path(sd_token)

        if img_path not in images:
            images.append(img_path)
        if img_path not in annotations:
            annotations[img_path] = []

        left = int(box.translation[0] - box.size[0] / 2)
        top = int(box.translation[1] - box.size[1] / 2)
        bounds = Bounds2D(left, top, box.size[0], box.size[1])
        class_index = classes.index(box.detection_name)
        annotations[img_path].append(Object2D(bounds, class_index))

    return Dataset(DATASET_NAME, images, classes, annotations)
