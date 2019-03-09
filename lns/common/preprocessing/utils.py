import random

from typing import Dict, List, Tuple

from lns.common.dataset import Dataset
from lns.common import config


def set_proportions(name: str,
                    images: Dict[str, List[str]], classes: List[str],
                    annotations: Dict[str, List[Dict[str, int]]],
                    proportion: float) -> Dataset:

    if proportion == 1.0:
        return Dataset(name, images, classes, annotations)

    new_annotations = {}

    print(images)

    for key, val in images.items():
        total = int(len(val) * proportion)

        random.seed(config.RAND_SEED)
        indices = random.sample([i for i in range(int(len(val)))], total)

        images[key] = [images[key][i] for i in indices]

        for path in images[key]:
            new_annotations[path] = annotations[path]

    return Dataset(name, images, classes, new_annotations)


def create_testset(name: str,
                   images: Dict[str, List[str]], classes: List[str],
                   annotations: Dict[str, List[Dict[str, int]]],
                   proportion: float) -> Dataset:

    if proportion == 1.0:
        return Dataset(name, images, classes, annotations)

    new_annotations = {}

    for key, val in images.items():
        total = int(len(val) * proportion)

        random.seed(config.RAND_SEED)
        inv_indices = sorted(random.sample(
            [i for i in range(int(len(val)))], total
        ))

        indices = []
        for i in range(int(len(val))):
            if len(inv_indices) != 0:
                if i == inv_indices[0]:
                    del inv_indices[0]
            else:
                indices.append(i)

        images[key] = [images[key][i] for i in indices]

        for path in images[key]:
            new_annotations[path] = annotations[path]

    return Dataset(name, images, classes, new_annotations)

def delete_empties(annotations: Dict[str, List[Dict[str, int]]],
                    images: Dict[str, List[str]]) -> Tuple:
    delete_keys = []
    for key, anno in annotations.items():
        if len(anno) == 0:
            print("deleting {}".format(key))
            delete_keys.append(key)

    for d in delete_keys:
        del annotations[d]
        del images[images.index(d)]

    return annotations, images