"""YOLOv3 trainer.

The module manages the representation of a YOLOv3 training session along with all associated data.
"""
import dataclasses
import os
import random
import shutil
from typing import Optional, Union

import tensorflow as tf  # type: ignore

from lns.common import config
from lns.common.dataset import Dataset
from lns.common.train import Trainer
from lns.yolo._lib.get_kmeans import get_kmeans, parse_anno
from lns.yolo.model import YoloModel
from lns.yolo.process import YoloData, YoloProcessor
from lns.yolo.settings import YoloSettings
from lns.yolo._lib.utils.misc_utils import parse_anchors, read_class_names
from lns.yolo._lib.utils.nms_utils import gpu_nms
from lns.yolo._lib.model import yolov3


class YoloTrainer(Trainer[YoloModel, YoloData, YoloSettings]):
    """Manages the YOLOv3 training environment and execution.

    Contains and encapsulates all training setup and files under one namespace.
    """

    SUBPATHS = {
        "log_folder": Trainer.Subpath(
            path="log", temporal=False, required=True, path_type=Trainer.PathType.FOLDER),
        "checkpoint_folder": Trainer.Subpath(
            path="checkpoint", temporal=False, required=True, path_type=Trainer.PathType.FOLDER),
        "anchors_file": Trainer.Subpath(
            path="anchors", temporal=False, required=False, path_type=Trainer.PathType.FILE),
        "progress_file": Trainer.Subpath(
            path="progress", temporal=False, required=False, path_type=Trainer.PathType.FILE),
        "frozen_graph_file": Trainer.Subpath(
            path="frozen_graph.pb", temporal=False, required=False, path_type=Trainer.PathType.FILE),
        "train_annotations_file": Trainer.Subpath(
            path="train_annotations", temporal=True, required=False, path_type=Trainer.PathType.FILE),
        "val_annotations_file": Trainer.Subpath(
            path="val_annotations", temporal=True, required=False, path_type=Trainer.PathType.FILE),
        "classes_file": Trainer.Subpath(
            path="classes", temporal=False, required=False, path_type=Trainer.PathType.FILE),
    }

    INITIAL_WEIGHTS_NAME = "yolov3.ckpt"
    INITIAL_WEIGHTS = os.path.join(config.RESOURCES_ROOT, config.WEIGHTS_FOLDER_NAME, INITIAL_WEIGHTS_NAME)

    settings: YoloSettings

    def __init__(self, name: str, dataset: Optional[Union[str, Dataset]] = None, load: bool = True) -> None:
        """Initialize a YOLOv3 trainer with the given unique <name>.

        Sources data from the <dataset> given, if any.
        If <load> is set to False removes any existing training files before training.
        """
        super().__init__(name, dataset,
                         _processor=YoloProcessor, _settings=YoloSettings,
                         _load=load, _subpaths=YoloTrainer.SUBPATHS)

    @property
    def model(self) -> Optional[YoloModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        weights = self.get_weights_path()
        anchors = self._paths["anchors_file"]
        classes = self._paths["classes_file"]

        model = None
        if all(os.path.exists(path) for path in (anchors, classes)):
            model = YoloModel(weights, anchors, classes, self.settings)
        return model

    def get_weights_path(self) -> str:
        """Get the path to most up-to-date weights associated with this trainer."""
        checkpoints = os.listdir(self._paths["checkpoint_folder"])
        if "checkpoint" in checkpoints:
            checkpoints.remove("checkpoint")
        if checkpoints:
            return os.path.join(
                self._paths["checkpoint_folder"],
                max(checkpoints, key=lambda checkpoint: int(checkpoint.split("_")[3])).rsplit(".", 1)[0])
        return YoloTrainer.INITIAL_WEIGHTS

    def train(self, settings: Optional[YoloSettings] = None) -> None:
        """Begin training the model."""
        settings = settings if settings else self._load_settings()
        self._set_settings(settings)
        self._generate_split()
        self._generate_anchors()
        self._generate_classes()
        weights_path = self.settings.initial_weights if self.settings.initial_weights else self.get_weights_path()

        from lns.yolo._lib import args
        args.train_file = self._paths["train_annotations_file"]
        args.val_file = self._paths["val_annotations_file"]
        args.restore_path = weights_path
        args.save_dir = self._paths["checkpoint_folder"] + "/"
        args.log_dir = self._paths["log_folder"]
        args.progress_log_path = self._paths["progress_file"]
        args.anchor_path = self._paths["anchors_file"]
        args.class_name_path = self._paths["classes_file"]
        args.global_step = self._get_global_step(weights_path)
        for field, setting in dataclasses.asdict(self.settings).items():
            setattr(args, field, setting)
        args.init()

        # Importing train will begin training
        try:
            from lns.yolo._lib import train  # pylint:disable=unused-import  # noqa
        except KeyboardInterrupt:
            print("Training interrupted")
        else:
            print("Training completed succesfully")

    # pylint: disable=too-many-locals
    # pylint: disable=E1101
    def export_graph(self) -> None:
        """Export a frozen graph in .pb format to the specified path."""
        anchors = parse_anchors(self._paths["anchors_file"])
        classes = read_class_names(self._paths["classes_file"])
        num_class = len(classes)

        with tf.Session() as sess:
            # build graph
            input_data = tf.placeholder(tf.float32, [None, None, None, 3], name='input')
            yolo_model = yolov3(num_class, anchors, use_static_shape=False)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(input_data, False)
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
            pred_scores = pred_confs * pred_probs
            _, _, _ = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=20, score_thresh=0.5, nms_thresh=0.5)
            # restore weight
            saver = tf.train.Saver()
            saver.restore(sess, self.get_weights_path())
            # save
            output_node_names = [
                "output/boxes",
                "output/scores",
                "output/labels",
                "input",
            ]

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                output_node_names
            )

            with tf.gfile.GFile(self._paths['frozen_graph_file'], "wb") as new_file:
                new_file.write(output_graph_def.SerializeToString())

            print("{} ops written to {}.".format(len(output_graph_def.node), self._paths['frozen_graph_file']))  # noqa

    def _generate_anchors(self) -> None:
        print("Generating anchors...")
        annotations = parse_anno(self.data.get_annotations(), self.settings.img_size)
        anchors, _ = get_kmeans(annotations, self.settings.num_clusters)
        anchors_string = ", ".join(f"{anchor[0]},{anchor[1]}" for anchor in anchors)
        with open(self._paths["anchors_file"], "w") as anchors_file:
            anchors_file.write(anchors_string)
        print("Anchors generated.")

    def _generate_split(self) -> None:
        annotations_file = self.data.get_annotations()
        with open(annotations_file, "r") as file:
            lines = file.readlines()
        random.shuffle(lines)

        val_split = int(self.settings.val_split * len(lines))
        for i, line in enumerate(lines):
            split_line = line.split()
            split_line[0] = str(i if i < val_split else (i - val_split))
            lines[i] = " ".join(split_line)

        val_lines = lines[:val_split]
        train_lines = lines[val_split:]

        with open(self._paths["val_annotations_file"], "w") as val_file:
            for line in val_lines:
                val_file.write(line + "\n")
        with open(self._paths["train_annotations_file"], "w") as train_file:
            for line in train_lines:
                train_file.write(line + "\n")

    def _generate_classes(self) -> None:
        shutil.copy(self.data.get_classes(), self._paths["classes_file"])

    # pylint: disable=no-self-use
    def _get_global_step(self, checkpoint_path: str) -> int:
        print("Restoring global step from checkpoint file name...")
        name = os.path.basename(checkpoint_path)

        # Example model checkpoint name: model-epoch_60_step_43309_loss_0.3424_lr_1e-05
        try:
            step = int(name.split("_")[3])
            print(f"Determined global step to be {step}.")
            return step
        except (IndexError, ValueError, TypeError):
            print("Could not determine global step from checkpoint file name. Defaulting to zero.")
            return 0
