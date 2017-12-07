"""
This is the code for running the whole tiny-yolo model.

Basically the algorithm is:
1. Open an image or video and reshape to (416, 416, 3)
2. Forward propagate the image through tiny-yolo model
1. Convert raw yolo output to readable form: boxes, scores, classes (and mark box corners for each box)
2. Filter out boxes with low probability of present object
3. Remove overlapping boxes using non-max-suppression
4. Show the result
"""

from build_model import model
from utils import process_yolo_output, filter_yolo_boxes, non_max_suppression, object_list, get_colors
from utils import rescale_coordinates_back_to_original_image
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2


def test_on_video(filter_threshold, iou_threshold, video_source):
    """
    Parameters:
    filter_threshold - class score threshold
    iou_threshold - intersection over union threshold
    video_source - 0 for cam or source to video file (relative to this wd)

    Returns:
    video stream (exit stream with ESC!)
    """
    # TODO: specify a video file or 0 for pc camera stream
    cam = cv2.VideoCapture(video_source)
    colors = get_colors()
    while True:

        # read frames
        stream, frame = cam.read()
        height, width, _ = frame.shape
        resized_frame = cv2.resize(frame, (416, 416)).reshape([1, 416, 416, 3]) / 255.

        # forward-prop yolo
        yolo_output = Model.predict(resized_frame)
        boxes, scores, classes = process_yolo_output(yolo_output)
        boxes, scores, classes = filter_yolo_boxes(boxes, scores, classes, threshold=filter_threshold)
        boxes, scores, classes = non_max_suppression(boxes, scores, classes, threshold=iou_threshold)
        boxes = rescale_coordinates_back_to_original_image(boxes, height, width)

        # draw
        for i in range(len(boxes)):

            # get attributes of an object
            x1, y1, x2, y2 = boxes[i][3], boxes[i][0], boxes[i][1], boxes[i][2]
            object_class, object_score = object_list[classes[i]], scores[i]
            object_color = colors[classes[i]]

            # mark the object
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=object_color, thickness=2)
            cv2.putText(frame, '{} {:.2f}'.format(object_class, object_score),
                        (x2, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        color=object_color,
                        fontScale=1)

        # stream video
        cv2.imshow('web-cam', frame)
        if cv2.waitKey(1) == 27:  # esc
            cam.release()
            cv2.destroyAllWindows()
            break


def test_on_image(filter_threshold, iou_threshold, image_source, output_dir):
    """
    Parameters:
    filter_threshold - class score threshold
    iou_threshold - intersection over union threshold
    image_source - source to image file (relative to this wd)

    Returns:
    shows processed image
    """
    # TODO: specify input image path
    im = Image.open(os.getcwd() + image_source)
    width, height = im.size
    im_resized = im.resize((416, 416), Image.ANTIALIAS)
    yolo_input = np.asarray(im_resized).reshape(1, 416, 416, 3) / 255

    # forward-prop and process output
    yolo_output = Model.predict(yolo_input)
    boxes, scores, classes = process_yolo_output(yolo_output)
    boxes, scores, classes = filter_yolo_boxes(boxes, scores, classes, threshold=filter_threshold)
    boxes, scores, classes = non_max_suppression(boxes, scores, classes, threshold=iou_threshold)
    boxes = rescale_coordinates_back_to_original_image(boxes, height, width)

    # draw output
    draw = ImageDraw.Draw(im)
    line_width = 5
    colors = get_colors()

    for i in range(len(boxes)):

        x1, y1, x2, y2 = boxes[i][3], boxes[i][0], boxes[i][1], boxes[i][2]

        # draw multiple lines to increase its width
        for line in range(line_width):
            draw.rectangle([x2 - 1 * line, y1 + 1 * line, x1 + 1 * line, y2 - 1 * line],
                           fill=None, outline=colors[classes[i]])

            draw.text((x2, y1), '{} {:.2f}'.format(object_list[classes[i]], scores[i]),
                      font=ImageFont.truetype('arial', 16), fill=colors[classes[i]])

    # TODO: specify output image path and name
    im.save('output_image', "JPEG")
    output_image = Image.open(os.getcwd() + output_dir)
    output_image.show()


# main
Model = model()
test_on_image(filter_threshold=0.4, iou_threshold=0.9, image_source='\\images\\image0.jpg', output_dir='\\output_image')
test_on_video(filter_threshold=0.3, iou_threshold=0.8, video_source=0)
