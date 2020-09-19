"""lyftperception dataset preprocessor.

The name has an appended _ to avoid a name conflict with the package.
"""

# from itertools import chain
# import matplotlib.pyplot as plt
import os
from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor
from lns.common.preprocessing.lyft_dataset_sdk import LyftDataset
from lns.common.preprocessing.lyft_dataset_sdk.utils.geometry_utils import view_points


DATASET_NAME = "LyftPerception"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _lyftperception(path: str = '/home/lns/.lns-training/resources/data/LyftPerception/train', datatype: str ='train') -> Dataset:  # noqa
    images: Dataset.Images = []
    classes: Dataset.Classes = ['car', 'pedestrian', 'animal', 'other_vehicle', 'bus',
                                'motorcycle', 'truck', 'emergency_vehicle', 'bicycle']
    annotations: Dataset.Annotations = {}

    lyft = LyftDataset(data_path=path,  # '/home/lns/.lns-training/resources/data/LyftPerception/train',
                    json_path=os.path.join(path, datatype + '_data'),
                        verbose=True, map_resolution = 0.1)
    # '/home/lns/.lns-training/resources/data/LyftPerception/train/train_data',
    
    for sample in lyft.sample:
        cam_token = sample['data']['CAM_FRONT']

        # Returns the data path as well as all annotations related to that sample_data.
        # Note that the boxes are transformed into the current sensor's coordinate frame.
        data_path, boxes, camera_intrinsic = lyft.get_sample_data(cam_token)
        images.append(str(data_path))
        annotations[str(data_path)] = []

        for box in boxes:
            img_corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
            # Take an outer rect of the 3d projection
            xmin = img_corners[0].min()
            xmax = img_corners[0].max()
            ymin = img_corners[1].min()
            ymax = img_corners[1].max()

            bounds = Bounds2D(xmin, ymin, xmax - xmin, ymax - ymin)
            # car, pedestrian, animal, other_vehicle, bus, motorcycle, truck, emergency_vehicle, bicycle
            label = box.name
            if label is not None:
                class_index = classes.index(box.name)  # noqa
                annotations[str(data_path)].append(Object2D(bounds, class_index))  # noqa

    return Dataset(DATASET_NAME, images, classes, annotations)
