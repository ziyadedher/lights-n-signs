"""nuScenes dataset preprocessor.

The name has an appended _ to avoid a name conflict with the package.
"""

from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor
from lns.common.preprocessing.nuscenes import NuScenes
from lns.common.preprocessing.nuscenes.eval.detection.constants import DETECTION_NAMES
from lns.common.preprocessing.nuscenes.eval.detection.utils import category_to_detection_name
from lns.common.preprocessing.nuscenes.utils.geometry_utils import view_points

# # unused imports from before
# from itertools import chain
# from lns.common.preprocessing.nuscenes.eval.common.loaders import load_gt
# import matplotlib.pyplot as plt

DATASET_NAME = "nuscenes"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _nuscenes(path: str) -> Dataset:  # noqa
    images: Dataset.Images = []
    classes: Dataset.Classes = DETECTION_NAMES
    annotations: Dataset.Annotations = {}

    nusc = NuScenes(version="v1.0-trainval", dataroot=path)

    for sample in nusc.sample:
        cam_token = sample['data']['CAM_FRONT']

        # Returns the data path as well as all annotations related to that sample_data.
        # Note that the boxes are transformed into the current sensor's coordinate frame.
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam_token)
        images.append(data_path)
        annotations[data_path] = []

        for box in boxes:
            img_corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
            # Take an outer rect of the 3d projection
            xmin = img_corners[0].min() if img_corners[0].min() >= 0 else 0  # 1224x1024
            xmax = img_corners[0].max() if img_corners[0].max() <= 1600 else 1600
            ymin = img_corners[1].min() if img_corners[1].min() >= 0 else 0
            ymax = img_corners[1].max() if img_corners[1].max() <= 900 else 900

            bounds = Bounds2D(xmin, ymin, xmax - xmin, ymax - ymin)
            label = category_to_detection_name(str(box.name))
            if label is not None:
                class_index = classes.index(str(category_to_detection_name(str(box.name))))
                annotations[data_path].append(Object2D(bounds, class_index))

    return Dataset(DATASET_NAME, images, classes, annotations)
