import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

from config import lns_squeezedet_config
from train import _draw_box
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', sys.argv[1],
    """Path to the checkpoint's yaml file""")
tf.app.flags.DEFINE_string(
    'input_path', sys.argv[2],
    """Input image to be detected""")
tf.app.flags.DEFINE_string(
    'out_dir', os.path.dirname(sys.argv[2]), """Directory to dump output image or video.""")


def image_benchmark():
  """Detect image."""

  with tf.Graph().as_default():
    mc = lns_squeezedet_config()
    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDet(mc, FLAGS.gpu)
      
    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(FLAGS.checkpoint)))

      for f in glob.iglob(FLAGS.input_path):
        im = cv2.imread(f)
        im = im.astype(np.float32, copy=False)
        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        input_image = im - mc.BGR_MEANS

        # Detect
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input:[input_image]})

        # Filter
        final_boxes, final_probs, final_class = model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        # TODO(bichen): move this color dict to configuration file
        cls2clr = {
            'car': (255, 191, 0),
            'cyclist': (0, 191, 255),
            'pedestrian':(255, 0, 191)
        }

        # Draw boxes
        _draw_box(
            im, final_boxes,
            [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(final_class, final_probs)],
            cdict=cls2clr,
        )

        file_name = os.path.split(f)[1]
        out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
        cv2.imwrite(out_file_name, im)
        print ('Image detection output saved to {}'.format(out_file_name))


def main(argv=None):
  image_benchmark()

if __name__ == '__main__':
    tf.app.run()
