"""Scale data preprocessor."""

import os
import urllib.request
from urllib.error import HTTPError
import cgi
from typing import List, Union
import requests
import scaleapi  # type: ignore

from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor


SCALE_API_KEY = 'live_c51c4273f60f4bcb9e86578c372aa51d'
HEADERS = {"Content-Type": "application/json"}

MAX_TO_PROCESS = 999999

LIGHTS_DATASET_NAME = "ScaleLights"
NEW_LIGHTS_YOUTUBE_NAME = "ScaleLights_New_Youtube"
NEW_LIGHTS_UTIAS_NAME = "ScaleLights_New_Utias"
NEW_LIGHTS_TRC_NAME = "ScaleLights_New_TRC"
NEW_LIGHTS_TEST_NAME = "ScaleLights_New_Test"
SIGNS_DATASET_NAME = "ScaleSigns"
OBJECTS_DATASET_NAME = "ScaleObjects"


def _scale_common(name: str, path: str, project: str, batch: Union[str, List[str]] = None) -> Dataset:  # noqa
    images: Dataset.Images = []
    classes: Dataset.Classes = []
    annotations: Dataset.Annotations = {}

    scale_data_path = os.path.join(path, 'images')

    available_batches = requests.get(
        "https://api.scale.com/v1/batches?project={}".format(project),
        headers=HEADERS,
        auth=(SCALE_API_KEY, '')).json()

    batch_names = [b['name'] for b in available_batches['docs'] if b['status'] == 'completed']
    if batch or batch == '' and isinstance(batch, str) and batch not in batch_names + ['']:
        raise ValueError("Batch name {} does not exist".format(batch))

    if batch and isinstance(batch, list):
        for bat in batch:
            if bat not in batch_names:
                raise ValueError("Batch name {} does not exist".format(bat))

    client = scaleapi.ScaleClient(SCALE_API_KEY)

    if batch is None:
        batches_to_retrieve = batch_names
    elif isinstance(batch, str) or batch == '':
        batches_to_retrieve = [batch]
    else:
        batches_to_retrieve = batch

    for batch_name in batches_to_retrieve:
        proper_batch_name = batch_name if batch_name else 'default'
        batch_path = os.path.join(scale_data_path, proper_batch_name)

        count = 0
        offset = 0
        has_next_page = True
        needs_download = False

        if not os.path.exists(scale_data_path):
            os.makedirs(scale_data_path)

        if not os.path.exists(batch_path):
            os.makedirs(batch_path)
            needs_download = True

        while has_next_page:
            tasklist = client.tasks(status="completed",
                                    project=project,
                                    batch=batch_name,
                                    offset=offset)
            offset += 100

            for obj in tasklist:
                task_id = obj.param_dict['task_id']
                task = client.fetch_task(task_id)
                bbox_list = task.param_dict['response']['annotations']
                img_url = task.param_dict['params']['attachment']

                try:
                    if 'drive.google.com' in img_url:
                        remotefile = urllib.request.urlopen(img_url)
                        content = remotefile.info()['Content-Disposition']
                        _, params = cgi.parse_header(content)
                        local_path = os.path.join(batch_path, params["filename"])
                    else:
                        local_path = os.path.join(batch_path, img_url.rsplit('/', 1)[-1])

                    if needs_download or not os.path.isfile(local_path):
                        # Download the image
                        urllib.request.urlretrieve(img_url, local_path)

                except HTTPError as error:
                    print("Image {} failed to download due to HTTPError {}: {}".format(
                        img_url, error.code, error.reason))
                    continue
                except TypeError as error:
                    print("Image {} failed to download due to improper header: {}".format(
                        img_url, str(error)))
                    continue

                annotations[local_path] = []
                for bbox in bbox_list:
                    # Get the label of the detected object
                    detection_class = bbox['label']

                    # Calculate the position and dimensions of the bounding box
                    x_min = int(bbox['left'])  # x-coordinate of top left corner
                    y_min = int(bbox['top'])  # y-coordinate of top left corner
                    width = int(bbox['width'])  # width of the bounding box
                    height = int(bbox['height'])  # height of the bounding box

                    # Get the class index if it has already been registered
                    # otherwise register it and select the index
                    try:
                        class_index = classes.index(detection_class)
                    except ValueError:
                        class_index = len(classes)
                        classes.append(detection_class)

                    # Package the detection
                    annotations[local_path].append(
                        Object2D(Bounds2D(x_min, y_min, width, height), class_index))

                images.append(local_path)
                print("Processed {}\r".format(local_path), end="")
                count += 1

                if len(tasklist) < 100 or count > MAX_TO_PROCESS:
                    has_next_page = False

    return Dataset(name, images, classes, annotations)


@Preprocessor.register_dataset_preprocessor(NEW_LIGHTS_UTIAS_NAME)
def _scale_lights_new_utias(path: str) -> Dataset:  # noqa
    dataset = _scale_common(NEW_LIGHTS_UTIAS_NAME, path, "light_labeling", batch="autoronto-lights-01-25-2020")
    return dataset


@Preprocessor.register_dataset_preprocessor(NEW_LIGHTS_TRC_NAME)
def _scale_lights_new_trc(path: str) -> Dataset:  # noqa
    dataset = _scale_common(NEW_LIGHTS_TRC_NAME, path, "light_labeling", batch="autoronto-lights-year3kickoff")
    return dataset


@Preprocessor.register_dataset_preprocessor(NEW_LIGHTS_YOUTUBE_NAME)
def _scale_lights_new_youtube(path: str) -> Dataset:  # noqa
    full = _scale_common("full", path, "light_labeling", batch="autoronto-lights-boston-chicago")
    splits = full.split([0.9, 0.1])
    youtube_dataset = splits[0]
    youtube_dataset._name = NEW_LIGHTS_YOUTUBE_NAME  # noqa
    test_dataset = splits[1] + Preprocessor.preprocess(NEW_LIGHTS_TRC_NAME)
    test_dataset._name = NEW_LIGHTS_TEST_NAME  # noqa
    Preprocessor.cache_preprocessed_data(NEW_LIGHTS_TEST_NAME, test_dataset)
    return youtube_dataset


@Preprocessor.register_dataset_preprocessor(NEW_LIGHTS_TEST_NAME)
def _scale_lights_new_test(path: str) -> Dataset:  # noqa
    youtube = Preprocessor.preprocess(NEW_LIGHTS_YOUTUBE_NAME)  # noqa
    dataset = Preprocessor.preprocess(NEW_LIGHTS_TEST_NAME)
    return dataset


@Preprocessor.register_dataset_preprocessor(LIGHTS_DATASET_NAME)
def _scale_lights(path: str) -> Dataset:  # noqa
    dataset = \
        _scale_common("lights1", path, "light_labeling", batch=['lights_2', 'mcity_lights', 'multipart_lights_1']) + \
        _scale_common("lights2", path, "light_labeling_old", batch='')
    dataset._name = LIGHTS_DATASET_NAME  # noqa
    return dataset


@Preprocessor.register_dataset_preprocessor(SIGNS_DATASET_NAME)
def _scale_signs(path: str, batch: str = None) -> Dataset:  # noqa
    return _scale_common(SIGNS_DATASET_NAME, path, "sign_labeling", batch=batch)


@Preprocessor.register_dataset_preprocessor(OBJECTS_DATASET_NAME)
def _scale_objects(path: str, batch: str = None) -> Dataset:  # noqa
    return _scale_common(OBJECTS_DATASET_NAME, path, "object_labeling", batch=batch)
