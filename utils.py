"""
Helper functions for processing raw yolo output

Much much help from:
https://github.com/allanzelener/YAD2K/blob/master/yad2k/models/keras_yolo.py
https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
"""

import numpy as np
import colorsys
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def process_yolo_output(yolo_output):
    """
    Converts raw yolo output to readable objects.
    model_output shape (None, 13, 13, 425), (5+80)*5 == 425

    Arguments:
    model_output = final yolo layer

    Returns:
    boxes - box corners (4, 13, 13, 5, 1) respectively (4 corners, box, box, anchors)
    scores - probability that object is present in each box (13, 13, 5, 1)
    classes - probability of each class present in a box (13, 13, 5, 80)
    """

    # get output dims
    _, h, w, _ = np.shape(yolo_output)

    # structure yolo output
    yolo_output = np.reshape(yolo_output, [w, h, 5, 85])
    box_xy = sigmoid(yolo_output[..., :2])
    box_wh = np.exp(yolo_output[..., 2:4])
    scores = sigmoid(yolo_output[..., 4:5])
    classes = softmax(yolo_output[..., 5:])

    # get conv index
    conv_index = np.array([_ for _ in np.ndindex(w, h)])
    conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering
    conv_index = conv_index.reshape(1, h, w, 1, 2)

    # reshape anchors (the anchors are taken from .cfg file)
    anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]
    anchors_tensor = np.reshape(anchors, [1, 1, 1, 5, 2])

    # get coordinates to each grid and anchor
    box_xy = (box_xy + conv_index) / w
    box_wh = box_wh * anchors_tensor / [w, h]

    # get corners of each box
    box_mins = box_xy - (box_wh / 2)
    box_maxes = box_xy + (box_wh / 2)
    boxes = np.concatenate([box_mins[..., 1:2], box_mins[..., 0:1], box_maxes[..., 1:2], box_maxes[..., 0:1]])

    return boxes, scores, classes


def filter_yolo_boxes(boxes, scores, classes, threshold=.7):
    """
    Filters boxes given probability threshold.

    Arguments:
    threshold is float, to filter boxes according to class prob

    Returns:
    Filtered boxes, scores, classes
    """

    # get a class for each box
    box_scores = scores * classes
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)

    # apply threshold
    score_mask = box_class_scores > threshold
    boxes = boxes[:, score_mask, :]
    scores = box_class_scores[score_mask]
    classes = box_classes[score_mask]
    
    return boxes, scores, classes


def non_max_suppression(boxes, scores, classes, threshold=0.99):
    """
    Filter overlapping boxes.
    Arguments: boxes, scores, classes, threshold
    Returns: boxes, scores, classes

    Source:
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """

    x1 = np.array(boxes[0, ...])
    y1 = np.array(boxes[1, ...])
    x2 = np.array(boxes[2, ...])
    y2 = np.array(boxes[3, ...])

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indexes = np.argsort([item[0] for item in y2])

    pick = []
    while len(indexes) > 0:

        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[indexes[:last]])
        yy1 = np.maximum(y1[i], y1[indexes[:last]])
        xx2 = np.minimum(x2[i], x2[indexes[:last]])
        yy2 = np.minimum(y2[i], y2[indexes[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[indexes[:last]]

        # delete all indexes that do not conform threshold
        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > threshold)[0])))

    return boxes[:, pick, :], scores[pick], classes[pick]


def rescale_coordinates_back_to_original_image(boxes, height, width):

    """
    Rescale box coordinates back to original image
    Arguments:
    boxes after non-max-suppression
    height and width of an original image
    Returns:
    A list of new boxes with rescaled coordinates
    """

    new_boxes = []
    _, all_boxes, _ = np.shape(boxes)
    for box in range(all_boxes):
        top, left, bottom, right = boxes[:, box, :]
        top = int(top * height)
        left = int(left * width)
        bottom = int(bottom * height)
        right = int(right * width)
        new_boxes.append([top, left, bottom, right])

    return new_boxes


def get_colors():
    """
    Generate 80 rgb colors for each class
    Returns: a list of random 80 colors
    """
    # Generate colors for drawing bounding boxes
    # https://github.com/allanzelener/YAD2K/blob/master/test_yolo.py
    hsv_tuples = [(x / 80, 1., 1.) for x in range(80)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(2)  # Fixed seed for consistent colors across runs
    random.shuffle(colors)  # Shuffle colors to de-correlate adjacent classes
    random.seed(None)  # Reset seed to default
    return colors


object_list = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
               'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

