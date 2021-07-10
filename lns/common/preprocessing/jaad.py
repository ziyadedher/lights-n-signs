"""JAAD data preprocessor."""

import os

from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor
from lns.common.preprocessing.JAAD.jaad_data import JAAD  # type: ignore

DATASET_NAME = "JAADDataset"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _jaad_dataset(path: str) -> Dataset:  # pylint:disable=too-many-locals, too-many-branches
    images: Dataset.Images = []
    classes: Dataset.Classes = ['pedestrian', 'ped', 'people']
    annotations: Dataset.Annotations = {}

    data_params = {
        'fstride': 1,
        'sample_type': 'all',
        'subset': 'all_videos',  # 'high_visibility' (high + low res, high vis), 'default' (high res, high vis only)
        'data_split_type': 'default',
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0
    }

    jaad = JAAD(data_path=path)
    jaad_anns = jaad.generate_database()

    # get all video ids
    train_ids, _ = jaad.get_data_ids('train', data_params)
    val_ids, _ = jaad.get_data_ids('val', data_params)
    test_ids, _ = jaad.get_data_ids('test', data_params)
    video_ids = train_ids + val_ids + test_ids

    for vid in video_ids:
        for pid in jaad_anns[vid]['ped_annotations']:
            imgs = [os.path.join(jaad.jaad_path, 'images', vid, '{:05d}.png'.format(f)) for f in
                    jaad_anns[vid]['ped_annotations'][pid]['frames']]
            boxes = jaad_anns[vid]['ped_annotations'][pid]['bbox']

            for box, img in zip(boxes, imgs):

                bounds = Bounds2D(box[0], box[1], box[2] - box[0], box[3] - box[1])
                if 'pedestrian' in jaad_anns[vid]['ped_annotations'][pid]['old_id']:
                    class_index = classes.index('pedestrian')
                elif 'people' in jaad_anns[vid]['ped_annotations'][pid]['old_id']:
                    class_index = classes.index('people')
                else:
                    class_index = classes.index('ped')

                if img not in annotations:
                    images.append(img)
                    annotations[img] = [Object2D(bounds, class_index)]
                else:
                    annotations[img].append(Object2D(bounds, class_index))

    return Dataset(DATASET_NAME, images, classes, annotations)
