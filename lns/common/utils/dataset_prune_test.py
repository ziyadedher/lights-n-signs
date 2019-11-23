"""Simple demo to test dataset pruning."""

from collections import defaultdict

from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from lns.common import config
from lns.common.dataset import Dataset
from lns.common.utils.img_area import img_area


def distance(dataset: Dataset) -> Dict[str, List[Tuple[int, float]]]:
    """Return a mapping of images to a list of object distances."""
    distances: Dict[str, List[Tuple[int, float]]] = {}

    images = dataset.images
    annotations = dataset.annotations

    for image in images:
        distances[image] = []
        image_area = img_area(image)

        for detection in annotations[image]:
            distances[image].append((detection.class_index, detection.bounds.area / image_area))

    return distances


if __name__ == "__main__":
    from lns.common.preprocess import Preprocessor
    Preprocessor.init_cached_preprocessed_data()
    for dataset_name in config.POSSIBLE_DATASETS:
        if dataset_name in ["mocked", "ScaleSigns", "ScaleObjects"]:
            # TODO: ScaleSigns and ScaleObjects haven't been cached yet
            continue
        print(dataset_name)

        dataset = Preprocessor.preprocess(dataset_name).prune(config.THRESHOLDS[dataset_name])
        result = distance(dataset)

        # aggregate statistics over dataset by class
        dists: Dict[int, List[float]] = defaultdict(list)
        for img in result:
            for bb in result[img]:
                dists[bb[0]].append(bb[1])

        class_names = dataset.classes
        all_dists: List[float] = []
        for class_index in dists:
            print(class_names[class_index])
            dists[class_index] = np.array(dists[class_index])

            # set all nonpositive ratios to infimum of nonnegative ratios to allow log
            dists[class_index][dists[class_index] <= 0] = np.min(dists[class_index][dists[class_index] > 0])

            plt_name = "%s_%s.png" % (dataset_name, class_names[class_index])
            plt.hist(dists[class_index])
            plt.savefig(plt_name)
            plt.clf()

            plt_name = "log_%s_%s.png" % (dataset_name, class_names[class_index])
            plt.hist(np.log(dists[class_index]))
            plt.savefig(plt_name)
            plt.clf()

            all_dists.extend(list(dists[class_index]))

        plt_name = "%s.png" % (dataset_name)
        plt.hist(all_dists)
        plt.savefig(plt_name)
        plt.clf()

        plt_name = "log_%s.png" % (dataset_name)
        plt.hist(np.log(all_dists))
        plt.savefig(plt_name)
        plt.clf()
