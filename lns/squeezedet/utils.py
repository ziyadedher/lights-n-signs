# Project: squeezeDetOnKeras
# Filename: utils
# Author: Bichen Wu, Christopher Ehmann
# Date: 29.11.17
# Organisation: searchInk
# Email: christopher@searchink.com



"""Utility functions in keras backend."""

import numpy as np
import time
import tensorflow as tf
import tensorflow.keras.backend as K


def iou(box1, box2):
  """Compute the Intersection-Over-Union of two given boxes.
  Args:
    box1: array of 4 elements [cx, cy, width, height].
    box2: same as above
  Returns:
    iou: a float number in range [0, 1]. iou of the two boxes.
  """

  lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
      max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
  if lr > 0:
    tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
        max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    if tb > 0:
      intersection = tb*lr
      union = box1[2]*box1[3]+box2[2]*box2[3]-intersection

      return intersection/union

  return 0



def batch_iou(boxes, box):
  """Compute the Intersection-Over-Union of a batch of boxes with another
  box.
  Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
  Returns:
    ious: array of a float number in range [0, 1].
  """
  lr = np.maximum(
      np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
      np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
      0
  )
  tb = np.maximum(
      np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
      np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
      0
  )
  inter = lr*tb
  union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
  return inter/union





def nms(boxes, probs, threshold):
  """Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format)
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """

  order = probs.argsort()[::-1]
  keep = [True]*len(order)

  for i in range(len(order)-1):
    ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
    for j, ov in enumerate(ovps):
      if ov > threshold:
        keep[order[j+i+1]] = False
  return keep

# TODO(bichen): this is not equivalent with full NMS. Need to improve it.

def recursive_nms(boxes, probs, threshold, form='center'):
  """Recursive Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format) or [xmin, ymin, xmax, ymax]
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """

  assert form == 'center' or form == 'diagonal', \
      'bounding box format not accepted: {}.'.format(form)

  if form == 'center':
    # convert to diagonal format
    boxes = np.array([bbox_transform(b) for b in boxes])

  areas = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
  hidx = boxes[:, 0].argsort()
  keep = [True]*len(hidx)

  def _nms(hidx):
    order = probs[hidx].argsort()[::-1]

    for idx in range(len(order)):
      if not keep[hidx[order[idx]]]:
        continue
      xx2 = boxes[hidx[order[idx]], 2]
      for jdx in range(idx+1, len(order)):
        if not keep[hidx[order[jdx]]]:
          continue
        xx1 = boxes[hidx[order[jdx]], 0]
        if xx2 < xx1:
          break
        w = xx2 - xx1
        yy1 = max(boxes[hidx[order[idx]], 1], boxes[hidx[order[jdx]], 1])
        yy2 = min(boxes[hidx[order[idx]], 3], boxes[hidx[order[jdx]], 3])
        if yy2 <= yy1:
          continue
        h = yy2-yy1
        inter = w*h
        iou = inter/(areas[hidx[order[idx]]]+areas[hidx[order[jdx]]]-inter)
        if iou > threshold:
          keep[hidx[order[jdx]]] = False

  def _recur(hidx):
    if len(hidx) <= 20:
      _nms(hidx)
    else:
      mid = len(hidx)/2
      _recur(hidx[:mid])
      _recur(hidx[mid:])
      _nms([idx for idx in hidx if keep[idx]])

  _recur(hidx)

  return keep

def sparse_to_dense(sp_indices, output_shape, values, default_value=0):
  """Build a dense matrix from sparse representations.
  Args:
    sp_indices: A [0-2]-D array that contains the index to place values.
    shape: shape of the dense matrix.
    values: A {0,1}-D array where values corresponds to the index in each row of
    sp_indices.
    default_value: values to set for indices not specified in sp_indices.
  Return:
    A dense numpy N-D array with shape output_shape.
  """

  assert len(sp_indices) == len(values), \
      'Length of sp_indices is not equal to length of values'

  array = np.ones(output_shape) * default_value
  for idx, value in zip(sp_indices, values):
    array[tuple(idx)] = value
  return array

def bgr_to_rgb(ims):
  """Convert a list of images from BGR format to RGB format."""
  out = []
  for im in ims:
    out.append(im[:,:,::-1])
  return out

def bbox_transform(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = cx-w/2
    out_box[1] = cy-h/2
    out_box[2] = cx+w/2
    out_box[3] = cy+h/2

    return out_box

def bbox_transform_inv(bbox):
    """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
    for numpy array or list of tensors.
    """
    xmin, ymin, xmax, ymax = bbox
    out_box = [[]]*4

    width       = xmax - xmin + 1.0
    height      = ymax - ymin + 1.0
    out_box[0]  = xmin + 0.5*width
    out_box[1]  = ymin + 0.5*height
    out_box[2]  = width
    out_box[3]  = height

    return out_box


def safe_exp(w, thresh):
  """Safe exponential function for tensors."""

  slope = np.exp(thresh)
  lin_bool = w > thresh

  lin_region = K.cast(lin_bool, dtype='float32')

  lin_out = slope*(w - thresh + 1.)

  exp_out = K.exp(K.switch(lin_bool, K.zeros_like(w), w))

  out = lin_region*lin_out + (1.-lin_region)*exp_out

  return out


def safe_exp_np(w, thresh):
  """Safe exponential function for numpy tensors."""

  slope = np.exp(thresh)
  lin_bool = w > thresh

  lin_region = lin_bool.astype(float)

  lin_out = slope*(w - thresh + 1.)

  exp_out = np.exp(np.where(lin_bool, np.zeros_like(w), w))

  out = lin_region*lin_out + (1.-lin_region)*exp_out

  return out




def boxes_from_deltas(pred_box_delta, config):
    """
    Converts prediction deltas to bounding boxes
    
    Arguments:
        pred_box_delta {[type]} -- tensor of deltas
        config {[type]} -- hyperparameter dict
    
    Returns:
        [type] -- tensor of bounding boxes
    """



    # Keras backend allows no unstacking

    delta_x = pred_box_delta[:, :, 0]
    delta_y = pred_box_delta[:, :, 1]
    delta_w = pred_box_delta[:, :, 2]
    delta_h = pred_box_delta[:, :, 3]

    # get the coordinates and sizes of the anchor boxes from config

    anchor_x = config.ANCHOR_BOX[:, 0]
    anchor_y = config.ANCHOR_BOX[:, 1]
    anchor_w = config.ANCHOR_BOX[:, 2]
    anchor_h = config.ANCHOR_BOX[:, 3]

    # as we only predict the deltas, we need to transform the anchor box values before computing the loss

    box_center_x = anchor_x + delta_x * anchor_w
    box_center_y = anchor_y + delta_y * anchor_h
    box_width = anchor_w * safe_exp(delta_w, config.EXP_THRESH)
    box_height = anchor_h * safe_exp(delta_h, config.EXP_THRESH)

    # tranform into a real box with four coordinates

    xmins, ymins, xmaxs, ymaxs = bbox_transform([box_center_x, box_center_y, box_width, box_height])

    # trim boxes if predicted outside

    xmins = K.minimum(
        K.maximum(0.0, xmins), config.IMAGE_WIDTH - 1.0)
    ymins = K.minimum(
        K.maximum(0.0, ymins), config.IMAGE_HEIGHT - 1.0)
    xmaxs = K.maximum(
        K.minimum(config.IMAGE_WIDTH - 1.0, xmaxs), 0.0)
    ymaxs = K.maximum(
        K.minimum(config.IMAGE_HEIGHT - 1.0, ymaxs), 0.0)

    det_boxes = K.permute_dimensions(
        K.stack(bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
        (1, 2, 0)
    )
    
    return (det_boxes)


def boxes_from_deltas_np(pred_box_delta, config):

    """
    Converts prediction deltas to bounding boxes, but in numpy
    
    Arguments:
        pred_box_delta {[type]} -- tensor of deltas
        config {[type]} -- hyperparameter dict
    
    Returns:
        [type] -- tensor of bounding boxes
    """


    # Keras backend allows no unstacking

    delta_x = pred_box_delta[:, :, 0]
    delta_y = pred_box_delta[:, :, 1]
    delta_w = pred_box_delta[:, :, 2]
    delta_h = pred_box_delta[:, :, 3]

    # get the coordinates and sizes of the anchor boxes from config

    anchor_x = config.ANCHOR_BOX[:, 0]
    anchor_y = config.ANCHOR_BOX[:, 1]
    anchor_w = config.ANCHOR_BOX[:, 2]
    anchor_h = config.ANCHOR_BOX[:, 3]

    # as we only predict the deltas, we need to transform the anchor box values before computing the loss

    box_center_x = anchor_x + delta_x * anchor_w
    box_center_y = anchor_y + delta_y * anchor_h
    box_width = anchor_w * safe_exp_np(delta_w, config.EXP_THRESH)
    box_height = anchor_h * safe_exp_np(delta_h, config.EXP_THRESH)

    # tranform into a real box with four coordinates

    xmins, ymins, xmaxs, ymaxs = bbox_transform([box_center_x, box_center_y, box_width, box_height])

    # trim boxes if predicted outside

    xmins = np.minimum(
        np.maximum(0.0, xmins), config.IMAGE_WIDTH - 1.0)
    ymins = np.minimum(
        np.maximum(0.0, ymins), config.IMAGE_HEIGHT - 1.0)
    xmaxs = np.maximum(
        np.minimum(config.IMAGE_WIDTH - 1.0, xmaxs), 0.0)
    ymaxs = np.maximum(
        np.minimum(config.IMAGE_HEIGHT - 1.0, ymaxs), 0.0)

    det_boxes = np.transpose(
        np.stack(bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
        (1, 2, 0)
    )

    return (det_boxes)

def slice_predictions(y_pred, config):
    """
    :param y_pred: network output
    :param config: config file
    :return: unpadded and sliced predictions
    """
    
    # calculate non padded entries
    n_outputs = config.CLASSES + 1 + 4
    # slice and reshape network output
    y_pred = y_pred[:, :, 0:n_outputs]
    y_pred = K.reshape(y_pred, (config.BATCH_SIZE, config.N_ANCHORS_HEIGHT, config.N_ANCHORS_WIDTH, -1))
    
    # number of class probabilities, n classes for each anchor
    
    num_class_probs = config.ANCHOR_PER_GRID * config.CLASSES

    # slice pred tensor to extract class pred scores and then normalize them
    pred_class_probs = K.reshape(
        K.softmax(
            K.reshape(
                y_pred[:, :, :, :num_class_probs],
                [-1, config.CLASSES]
            )
        ),
        [config.BATCH_SIZE, config.ANCHORS, config.CLASSES],
    )

    # number of confidence scores, one for each anchor + class probs
    num_confidence_scores = config.ANCHOR_PER_GRID + num_class_probs

    # slice the confidence scores and put them trough a sigmoid for probabilities
    pred_conf = K.sigmoid(
        K.reshape(
            y_pred[:, :, :, num_class_probs:num_confidence_scores],
            [config.BATCH_SIZE, config.ANCHORS]
        )
    )

    # slice remaining bounding box_deltas
    pred_box_delta = K.reshape(
        y_pred[:, :, :, num_confidence_scores:],
        [config.BATCH_SIZE, config.ANCHORS, 4]
    )
    
    return [pred_class_probs, pred_conf, pred_box_delta]


def slice_predictions_np(y_pred, config):
    """
    does the same as above, only uses numpy
    :param y_pred: network output
    :param config: config file
    :return: unpadded and sliced predictions
    """

    # calculate non padded entries
    n_outputs = config.CLASSES + 1 + 4
    # slice and reshape network output
    y_pred = y_pred[:, :, 0:n_outputs]
    y_pred = np.reshape(y_pred, (config.BATCH_SIZE, config.N_ANCHORS_HEIGHT, config.N_ANCHORS_WIDTH, -1))

    # number of class probabilities, n classes for each anchor

    num_class_probs = config.ANCHOR_PER_GRID * config.CLASSES

    # slice pred tensor to extract class pred scores and then normalize them
    pred_class_probs = np.reshape(
        softmax(
            np.reshape(
                y_pred[:, :, :, :num_class_probs],
                [-1, config.CLASSES]
            )
        ),
        [config.BATCH_SIZE, config.ANCHORS, config.CLASSES],
    )

    # number of confidence scores, one for each anchor + class probs
    num_confidence_scores = config.ANCHOR_PER_GRID + num_class_probs

    # slice the confidence scores and put them trough a sigmoid for probabilities
    pred_conf = sigmoid(
        np.reshape(
            y_pred[:, :, :, num_class_probs:num_confidence_scores],
            [config.BATCH_SIZE, config.ANCHORS]
        )
    )

    # slice remaining bounding box_deltas
    pred_box_delta = np.reshape(
        y_pred[:, :, :, num_confidence_scores:],
        [config.BATCH_SIZE, config.ANCHORS, 4]
    )

    return [pred_class_probs, pred_conf, pred_box_delta]


def tensor_iou(box1, box2, input_mask, config):
    """Computes pairwise IOU of two lists of boxes
    
    Arguments:
        box1 {[type]} -- First list of boxes
        box2 {[type]} -- Second list of boxes
        input_mask {[type]} -- Zero-One indicating which boxes to compute
        config {[type]} -- dict containing hyperparameters
    
    Returns:
        [type] -- [description]
    """

    
    xmin = K.maximum(box1[0], box2[0])
    ymin = K.maximum(box1[1], box2[1])
    xmax = K.minimum(box1[2], box2[2])
    ymax = K.minimum(box1[3], box2[3])

    w = K.maximum(0.0, xmax - xmin)
    h = K.maximum(0.0, ymax - ymin)

    intersection = w * h

    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    union = w1 * h1 + w2 * h2 - intersection

    return intersection / (union + config.EPSILON) * K.reshape(input_mask, [config.BATCH_SIZE, config.ANCHORS])



def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - np.max(x))
    return e_x / np.expand_dims(np.sum(e_x,axis=axis), axis=axis)


def sigmoid(x):
    """Sigmoid function
    
    Arguments:
        x {[type]} -- input
    
    Returns:
        [type] -- sigmoid(x)
    """


    return 1/(1+np.exp(-x))

  # Project: squeezeDetOnKeras
# Filename: dataGenerator
# Author: Christopher Ehmann
# Date: 29.11.17
# Organisation: searchInk
# Email: christopher@searchink.com

import threading
import cv2
import numpy as np
import random
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

#we could maybe use the standard data generator from keras?
def read_image_and_gt(img_files, gt_files, config):
    '''
    Transform images and send transformed image and label
    :param img_files: list of image files including the path of a batch
    :param gt_files: list of gt files including the path of a batch
    :param config: config dict containing various hyperparameters
    :return images and annotations
    '''

    labels = []
    bboxes  = []
    deltas = []
    aidxs  = []


    #loads annotations from file
    def load_annotation(gt_file):

        with open(gt_file, 'r') as f:
            lines = f.readlines()
        f.close()

        annotations = []

        #each line is an annotation bounding box
        for line in lines:
            obj = line.strip().split(' ')

            #get class, if class is not in listed, skip it
            try:
                cls = config.CLASS_TO_IDX[obj[0].lower().strip()]
                # print cls


                #get coordinates
                xmin = float(obj[4])
                ymin = float(obj[5])
                xmax = float(obj[6])
                ymax = float(obj[7])


                #check for valid bounding boxes
                assert xmin >= 0.0 and xmin <= xmax, \
                    'Invalid bounding box x-coord xmin {} or xmax {} at {}' \
                        .format(xmin, xmax, gt_file)
                assert ymin >= 0.0 and ymin <= ymax, \
                    'Invalid bounding box y-coord ymin {} or ymax {} at {}' \
                        .format(ymin, ymax, gt_file)


                #transform to  point + width and height representation
                x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])

                annotations.append([x, y, w, h, cls])

            except:
                continue
        return annotations

    #init tensor of images
    imgs = np.zeros((config.BATCH_SIZE, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.N_CHANNELS))

    img_idx = 0

    #iterate files
    for img_name, gt_name in zip(img_files, gt_files):


        #open img
        img = cv2.imread(img_name).astype(np.float32, copy=False)


        # scale image
        img = cv2.resize( img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

        #subtract means
        img = (img - np.mean(img))/ np.std(img)

        #store original height and width?
        orig_h, orig_w, _ = [float(v) for v in img.shape]


        #print(orig_h, orig_w)
        # load annotations
        annotations = load_annotation(gt_name)
        if len(annotations) == 0: continue 
        
        #split in classes and boxes
        labels_per_file = [a[4] for a in annotations]


        bboxes_per_file = np.array([a[0:4] for a in annotations])

        #and store
        imgs[img_idx] = np.asarray(img)
        
        img_idx += 1


        # scale annotation
        x_scale = config.IMAGE_WIDTH / orig_w
        y_scale = config.IMAGE_HEIGHT / orig_h


        #scale boxes
        bboxes_per_file[:, 0::2] = bboxes_per_file[:, 0::2] * x_scale
        bboxes_per_file[:, 1::2] = bboxes_per_file[:, 1::2] * y_scale


        bboxes.append(bboxes_per_file)

        aidx_per_image, delta_per_image = [], []
        aidx_set = set()


        #iterate all bounding boxes for a file
        for i in range(len(bboxes_per_file)):


            #compute overlaps of bounding boxes and anchor boxes
            overlaps = batch_iou(config.ANCHOR_BOX, bboxes_per_file[i])


            #achor box index
            aidx = len(config.ANCHOR_BOX)

            #sort for biggest overlaps
            for ov_idx in np.argsort(overlaps)[::-1]:
                #when overlap is zero break
                if overlaps[ov_idx] <= 0:
                    break
                #if one is found add and break
                if ov_idx not in aidx_set:
                    aidx_set.add(ov_idx)
                    aidx = ov_idx
                    break

            # if the largest available overlap is 0, choose the anchor box with the one that has the
            # smallest square distance
            if aidx == len(config.ANCHOR_BOX):
                dist = np.sum(np.square(bboxes_per_file[i] - config.ANCHOR_BOX), axis=1)
                for dist_idx in np.argsort(dist):
                    if dist_idx not in aidx_set:
                        aidx_set.add(dist_idx)
                        aidx = dist_idx
                        break


            #compute deltas for regression
            box_cx, box_cy, box_w, box_h = bboxes_per_file[i]
            delta = [0] * 4
            delta[0] = (box_cx - config.ANCHOR_BOX[aidx][0]) / config.ANCHOR_BOX[aidx][2]
            delta[1] = (box_cy - config.ANCHOR_BOX[aidx][1]) / config.ANCHOR_BOX[aidx][3]
            delta[2] = np.log(box_w / config.ANCHOR_BOX[aidx][2])
            delta[3] = np.log(box_h / config.ANCHOR_BOX[aidx][3])

            aidx_per_image.append(aidx)
            delta_per_image.append(delta)

        deltas.append(delta_per_image)
        aidxs.append(aidx_per_image)
        labels.append(labels_per_file)


    #we need to transform this batch annotations into a form we can feed into the model
    label_indices, bbox_indices, box_delta_values, mask_indices, box_values, \
          = [], [], [], [], []

    aidx_set = set()


    #iterate batch
    for i in range(len(labels)):
        #and annotations
        for j in range(len(labels[i])):
            if (i, aidxs[i][j]) not in aidx_set:
                aidx_set.add((i, aidxs[i][j]))
                label_indices.append(
                    [i, aidxs[i][j], labels[i][j]])
                mask_indices.append([i, aidxs[i][j]])
                bbox_indices.extend(
                    [[i, aidxs[i][j], k] for k in range(4)])
                box_delta_values.extend(deltas[i][j])
                box_values.extend(bboxes[i][j])


    #transform them into matrices
    input_mask =  np.reshape(
            sparse_to_dense(
                mask_indices,
                [config.BATCH_SIZE, config.ANCHORS],
                [1.0] * len(mask_indices)),

            [config.BATCH_SIZE, config.ANCHORS, 1])

    box_delta_input =  sparse_to_dense(
            bbox_indices, [config.BATCH_SIZE, config.ANCHORS, 4],
            box_delta_values)

    box_input =  sparse_to_dense(
            bbox_indices, [config.BATCH_SIZE, config.ANCHORS, 4],
            box_values)

    labels = sparse_to_dense(
            label_indices,
            [config.BATCH_SIZE, config.ANCHORS, config.CLASSES],
            [1.0] * len(label_indices))


    #concatenate ouputs
    Y = np.concatenate((input_mask, box_input,  box_delta_input, labels), axis=-1).astype(np.float32)



    return imgs, Y












def read_image_and_gt_with_original(img_files, gt_files, config):
    '''
    Transform images and send transformed image and label, but also return the image only resized
    :param img_files: list of image files including the path of a batch
    :param gt_files: list of gt files including the path of a batch
    :param config: config dict containing various hyperparameters
    :return images and annotations
    '''

    labels = []
    bboxes  = []
    deltas = []
    aidxs  = []


    #loads annotations from file
    def load_annotation(gt_file):

        with open(gt_file, 'r') as f:
            lines = f.readlines()
        f.close()

        annotations = []

        #each line is an annotation bounding box
        for line in lines:
            obj = line.strip().split(' ')

            #get class
            try:
                cls = config.CLASS_TO_IDX[obj[0].lower().strip()]
                # print cls


                #get coordinates
                xmin = float(obj[4])
                ymin = float(obj[5])
                xmax = float(obj[6])
                ymax = float(obj[7])


                #check for valid bounding boxes
                assert xmin >= 0.0 and xmin <= xmax, \
                    'Invalid bounding box x-coord xmin {} or xmax {} at {}' \
                        .format(xmin, xmax, gt_file)
                assert ymin >= 0.0 and ymin <= ymax, \
                    'Invalid bounding box y-coord ymin {} or ymax {} at {}' \
                        .format(ymin, ymax, gt_file)


                #transform to  point + width and height representation
                x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])

                annotations.append([x, y, w, h, cls])
            except:
                continue
        return annotations

    imgs = np.zeros((config.BATCH_SIZE, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.N_CHANNELS))
    imgs_only_resized = np.zeros((config.BATCH_SIZE, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.N_CHANNELS))

    img_idx = 0

    #iterate files
    for img_name, gt_name in zip(img_files, gt_files):


        #open img
        img = cv2.imread(img_name).astype(np.float32, copy=False)

        #store original height and width?
        orig_h, orig_w, _ = [float(v) for v in img.shape]
        
        # scale image
        img = cv2.resize( img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

        imgs_only_resized[img_idx] = img

        #subtract means
        img = (img - np.mean(img))/ np.std(img)




        #print(orig_h, orig_w)
        # load annotations
        annotations = load_annotation(gt_name)


        #split in classes and boxes
        labels_per_file = [a[4] for a in annotations]


        bboxes_per_file = np.array([a[0:4]for a in annotations])


        #TODO enable dynamic Data Augmentation
        """
        if config.DATA_AUGMENTATION:
            assert mc.DRIFT_X >= 0 and mc.DRIFT_Y > 0, \
                'mc.DRIFT_X and mc.DRIFT_Y must be >= 0'
            if mc.DRIFT_X > 0 or mc.DRIFT_Y > 0:
                # Ensures that gt boundibg box is not cutted out of the image
                max_drift_x = min(gt_bbox[:, 0] - gt_bbox[:, 2] / 2.0 + 1)
                max_drift_y = min(gt_bbox[:, 1] - gt_bbox[:, 3] / 2.0 + 1)
                assert max_drift_x >= 0 and max_drift_y >= 0, 'bbox out of image'
                dy = np.random.randint(-mc.DRIFT_Y, min(mc.DRIFT_Y + 1, max_drift_y))
                dx = np.random.randint(-mc.DRIFT_X, min(mc.DRIFT_X + 1, max_drift_x))
                # shift bbox
                gt_bbox[:, 0] = gt_bbox[:, 0] - dx
                gt_bbox[:, 1] = gt_bbox[:, 1] - dy
                # distort image
                orig_h -= dy
                orig_w -= dx
                orig_x, dist_x = max(dx, 0), max(-dx, 0)
                orig_y, dist_y = max(dy, 0), max(-dy, 0)
                distorted_im = np.zeros(
                    (int(orig_h), int(orig_w), 3)).astype(np.float32)
                distorted_im[dist_y:, dist_x:, :] = im[orig_y:, orig_x:, :]
                im = distorted_im
            # Flip image with 50% probability
            if np.random.randint(2) > 0.5:
                im = im[:, ::-1, :]
                gt_bbox[:, 0] = orig_w - 1 - gt_bbox[:, 0]
        """





        #and store
        imgs[img_idx] = np.asarray(img)
        #
        img_idx += 1


        # scale annotation
        x_scale = config.IMAGE_WIDTH / orig_w
        y_scale = config.IMAGE_HEIGHT / orig_h


        #scale boxes
        bboxes_per_file[:, 0::2] = bboxes_per_file[:, 0::2] * x_scale
        bboxes_per_file[:, 1::2] = bboxes_per_file[:, 1::2] * y_scale


        bboxes.append(bboxes_per_file)

        aidx_per_image, delta_per_image = [], []
        aidx_set = set()


        #iterate all bounding boxes for a file
        for i in range(len(bboxes_per_file)):


            #compute overlaps of bounding boxes and anchor boxes
            overlaps = batch_iou(config.ANCHOR_BOX, bboxes_per_file[i])


            #achor box index
            aidx = len(config.ANCHOR_BOX)

            #sort for biggest overlaps
            for ov_idx in np.argsort(overlaps)[::-1]:
                #when overlap is zero break
                if overlaps[ov_idx] <= 0:
                    break
                #if one is found add and break
                if ov_idx not in aidx_set:
                    aidx_set.add(ov_idx)
                    aidx = ov_idx
                    break

            # if the largest available overlap is 0, choose the anchor box with the one that has the
            # smallest square distance
            if aidx == len(config.ANCHOR_BOX):
                dist = np.sum(np.square(bboxes_per_file[i] - config.ANCHOR_BOX), axis=1)
                for dist_idx in np.argsort(dist):
                    if dist_idx not in aidx_set:
                        aidx_set.add(dist_idx)
                        aidx = dist_idx
                        break


            #compute deltas for regression
            box_cx, box_cy, box_w, box_h = bboxes_per_file[i]
            delta = [0] * 4
            delta[0] = (box_cx - config.ANCHOR_BOX[aidx][0]) / config.ANCHOR_BOX[aidx][2]
            delta[1] = (box_cy - config.ANCHOR_BOX[aidx][1]) / config.ANCHOR_BOX[aidx][3]
            delta[2] = np.log(box_w / config.ANCHOR_BOX[aidx][2])
            delta[3] = np.log(box_h / config.ANCHOR_BOX[aidx][3])

            aidx_per_image.append(aidx)
            delta_per_image.append(delta)

        deltas.append(delta_per_image)
        aidxs.append(aidx_per_image)
        labels.append(labels_per_file)


    #print(labels)

    #we need to transform this batch annotations into a form we can feed into the model
    label_indices, bbox_indices, box_delta_values, mask_indices, box_values, \
          = [], [], [], [], []

    aidx_set = set()


    #iterate batch
    for i in range(len(labels)):
        #and annotations
        for j in range(len(labels[i])):
            if (i, aidxs[i][j]) not in aidx_set:
                aidx_set.add((i, aidxs[i][j]))
                label_indices.append(
                    [i, aidxs[i][j], labels[i][j]])
                mask_indices.append([i, aidxs[i][j]])
                bbox_indices.extend(
                    [[i, aidxs[i][j], k] for k in range(4)])
                box_delta_values.extend(deltas[i][j])
                box_values.extend(bboxes[i][j])


    #transform them into matrices
    input_mask =  np.reshape(
            sparse_to_dense(
                mask_indices,
                [config.BATCH_SIZE, config.ANCHORS],
                [1.0] * len(mask_indices)),

            [config.BATCH_SIZE, config.ANCHORS, 1])

    box_delta_input =  sparse_to_dense(
            bbox_indices, [config.BATCH_SIZE, config.ANCHORS, 4],
            box_delta_values)

    box_input =  sparse_to_dense(
            bbox_indices, [config.BATCH_SIZE, config.ANCHORS, 4],
            box_values)

    labels = sparse_to_dense(
            label_indices,
            [config.BATCH_SIZE, config.ANCHORS, config.CLASSES],
            [1.0] * len(label_indices))


    #concatenate ouputs
    Y = np.concatenate((input_mask, box_input,  box_delta_input, labels), axis=-1).astype(np.float32)



    return imgs, Y, imgs_only_resized







@threadsafe_generator
def generator_from_data_path(img_names, gt_names, config, return_filenames=False, shuffle=False ):
    """
    Generator that yields (X, Y)
    :param img_names: list of images names with full path
    :param gt_names: list of gt names with full path
    :param config: config dict containing various hyperparameters
    :return: a generator yielding images and ground truths
   """


    assert len(img_names) == len(gt_names), "Number of images and ground truths not equal"

    if shuffle:
        #permutate images
        shuffled = list(zip(img_names, gt_names))
        random.shuffle(shuffled)
        img_names, gt_names = zip(*shuffled)

    """
    Each epoch will only process an integral number of batch_size
    but with the shuffling of list at the top of each epoch, we will
    see all training samples eventually, but will skip an amount
    less than batch_size during each epoch
    """


    nbatches, n_skipped_per_epoch = divmod(len(img_names), config.BATCH_SIZE)

    count = 1
    epoch = 0

    while 1:

        epoch += 1
        i, j = 0, config.BATCH_SIZE

        #mini batches within epoch
        mini_batches_completed = 0

        for _ in range(nbatches):
            #print(i,j)
            img_names_batch = img_names[i:j]
            gt_names_batch = gt_names[i:j]

            try:

                #get images and ground truths


                imgs, gts = read_image_and_gt(img_names_batch, gt_names_batch, config)

                mini_batches_completed += 1


                yield (imgs, gts)

            except IOError as err:

                count -= 1

            i = j
            j += config.BATCH_SIZE
            count += 1





def visualization_generator_from_data_path(img_names, gt_names, config, return_filenames=False, shuffle=False ):
    """
    Generator that yields (Images, Labels, unnormalized images)
    :param img_names: list of images names with full path
    :param gt_names: list of gt names with full path
    :param config
    :return:
   """


    assert len(img_names) == len(gt_names), "Number of images and ground truths not equal"

    """
    Each epoch will only process an integral number of batch_size
    # but with the shuffling of list at the top of each epoch, we will
    # see all training samples eventually, but will skip an amount
    # less than batch_size during each epoch
    """


    nbatches, n_skipped_per_epoch = divmod(len(img_names), config.BATCH_SIZE)

    count = 1
    epoch = 0

    while 1:

        epoch += 1
        i, j = 0, config.BATCH_SIZE

        #mini batches within epoch
        mini_batches_completed = 0

        for _ in range(nbatches):
            #print(i,j)
            img_names_batch = img_names[i:j]
            gt_names_batch = gt_names[i:j]

            try:

                #get images, ground truths and original color images
                imgs, gts, imgs_only_resized = read_image_and_gt_with_original(img_names_batch, gt_names_batch, config)

                mini_batches_completed += 1


                yield (imgs, gts, imgs_only_resized)

            except IOError as err:

                count -= 1

            i = j
            j += config.BATCH_SIZE
            count += 1