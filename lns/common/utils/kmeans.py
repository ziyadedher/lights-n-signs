"""Utility functions for computing k-means of bounding boxes to generate anchors."""

import numpy as np  # type: ignore


def kmeans(boxes, k, dist=np.median):
    """Calculate k-means clustering with the Intersection over Union (IoU) metric.

    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    np.random.seed()
    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters

    return clusters


def iou(box, clusters):
    """Calculate the Intersection over Union (IoU) between a box and k clusters.

    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x_min = np.minimum(clusters[:, 0], box[0])
    y_min = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x_min == 0) > 0 or np.count_nonzero(y_min == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x_min * y_min
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    return iou_


def avg_iou(boxes, clusters):
    """Calculate the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.

    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])
