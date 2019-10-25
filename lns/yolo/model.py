"""YOLOv3 model representation.

This module contains the prediction model that will be generated by the Haar training.
"""
from typing import Optional, List

import cv2               # type: ignore
import numpy as np       # type: ignore
import tensorflow as tf  # type: ignore

from lns.common.model import Model
from lns.common.structs import Object2D, Bounds2D
from lns.yolo.settings import YoloSettings

from lns.yolo._lib import args
from lns.yolo._lib.model import yolov3
from lns.yolo._lib.utils.nms_utils import gpu_nms
from lns.yolo._lib.utils.eval_utils import get_preds_gpu
from lns.yolo._lib.utils.data_aug import letterbox_resize


# FIXME: cannot create more than one YoloModel per Python instance because of TF name conflicts
class YoloModel(Model):
    """Detection model utilizing YOLOv3."""

    def __init__(self, weights_file: str, anchors_file: str, classes_file: str,
                 settings: Optional[YoloSettings] = None) -> None:
        """Initialize a YOLOv3 model."""
        if not settings:
            settings = YoloSettings()

        args.restore_path = weights_file
        args.anchor_path = anchors_file
        args.class_name_path = classes_file
        for field, setting in zip(settings._fields, settings):
            setattr(args, field, setting)
        args.init_inference()

        self._is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
        self._pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
        self._pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
        self._image = tf.placeholder(tf.float32, [1, args.img_size[1], args.img_size[0], 3])
        self._gpu_nms_op = gpu_nms(self._pred_boxes_flag, self._pred_scores_flag,
                                   args.class_num, args.nms_topk, args.score_threshold, args.nms_threshold)

        yolo_model = yolov3(args.class_num, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(self._image, is_training=self._is_training)
        self._y_pred = yolo_model.predict(pred_feature_maps)

        self._session = tf.Session()
        self._session.run([tf.global_variables_initializer()])
        tf.train.Saver().restore(self._session, args.restore_path)

    def predict(self, image: np.ndarray) -> List[Object2D]:  # pylint:disable=too-many-locals
        """Predict the required bounding boxes on the given <image>."""
        # TODO: implement non-letterbox resizing inference
        image, ratio, d_w, d_h = letterbox_resize(image, args.img_size[0], args.img_size[1], interp=1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0

        y_pred = self._session.run(self._y_pred, feed_dict={self._is_training: False, self._image: np.array([image])})
        pred_content = get_preds_gpu(self._session, self._gpu_nms_op, self._pred_boxes_flag,
                                     self._pred_scores_flag, [0], y_pred)

        predictions = []
        for pred in pred_content:
            _, x_min, y_min, x_max, y_max, score, label = pred
            x_min = int((x_min - d_w) / ratio)
            x_max = int((x_max - d_w) / ratio)
            y_min = int((y_min - d_h) / ratio)
            y_max = int((y_max - d_h) / ratio)

            predictions.append(Object2D(Bounds2D(x_min, y_min, x_max - x_min, y_max - y_min), label, score))
        return predictions
