"""

"""

from build_model import model
from PIL import Image
import os
import numpy as np

# get input
im = Image.open(os.getcwd() + '\\images\\image1.jpg')
im = im.resize((416, 416), Image.ANTIALIAS)
im.save('image0', 'JPEG')
x = np.asarray(im).reshape(1, 416, 416, 3) / 255

# get output
Model = model()
output = Model.predict(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=4, keepdims=True)


def process_yolo_output(model_output):
    """
    Converts raw yolo output to readable objects.
    model_output shape (None, 13, 13, 425), (5+80)*5 == 425

    Arguments:
    model_output = final yolo layer

    Returns:
    boxes - box coordinates (x, y, w ,h)
    scores - probability that object is present in each box (5 boxes)
    classes - probability of each class present in a box (80 classes)

    """

    # prepare raw model output
    b, h, w, num = np.shape(model_output)
    model_output_reshaped = model_output.reshape(b, h, w, 5, 85)
    scores = sigmoid(model_output_reshaped[..., 4:5])
    boxes = model_output_reshaped[..., 0:4]
    classes = softmax(model_output_reshaped[..., 5:])

    return boxes, scores, classes


def filter_boxes(boxes, scores, classes, threshold=0.6):
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
    boxes_filtered = boxes[score_mask]
    scores_filtered = box_class_scores[score_mask]
    classes_filtered = box_classes[score_mask]

    return boxes_filtered, scores_filtered, classes_filtered


def non_max_suppression(boxes, scores, classes, threshold=0.95):
    """
    Filter overlapping boxes.
    Arguments: boxes, scores, classes, threshold
    Returns: boxes, scores, classes

    Source:
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """

    x1 = np.array(boxes[..., 0])
    y1 = np.array(boxes[..., 1])
    x2 = np.array(boxes[..., 2])
    y2 = np.array(boxes[..., 3])

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    pick = []
    while len(idxs) > 0:

        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes that do not conform threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))

    return boxes[pick], scores[pick], classes[pick]


# boxes, scores, classes = process_yolo_output(output)
# boxes, scores, classes = filter_boxes(boxes, scores, classes, threshold=0.15)
# boxes, scores, classes = non_max_suppression(boxes, scores, classes, threshold=0.95)
#
#
# from PIL import Image, ImageFont, ImageDraw
#
# source_img = Image.open(os.getcwd() + '\\images\\image1.jpg')
#
# draw = ImageDraw.Draw(source_img)
# draw.rectangle(((0, 00), (100, 100)), fill=None)
# draw.text((20, 70), "something123", font=ImageFont.truetype('arial'))
#
# source_img.save('rect', "JPEG")
#
# xy = boxes[:, 0:2]
# wh = boxes[:, 2:4]
#
#
# left = (xy - (wh / 2.))
# right =