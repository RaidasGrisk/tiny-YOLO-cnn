"""
This is the code for running the whole thing.

What's being done here:
1. Open an image and reshape to (416, 416, 3)
2. Forward propagate the image through tiny-yolo model
1. Convert raw yolo output to readable form: boxes, scores, classes (and mark box corners for each box)
2. Filter out boxes with low probability of present object
3. Remove overlapping boxes using non-max-suppression
4. Draw the result
"""

from build_model import model
from utils import process_yolo_output, filter_yolo_boxes, non_max_suppression, object_list, colors
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# initiate model and get output
Model = model()

# get input
# TODO: specify input image path
im = Image.open(os.getcwd() + '\\images\\image0.jpg')
im_resized = im.resize((416, 416), Image.ANTIALIAS)
yolo_input = np.asarray(im_resized).reshape(1, 416, 416, 3) / 255

# forward-prop and process output
yolo_output = Model.predict(yolo_input)
boxes, scores, classes = process_yolo_output(yolo_output)
boxes, scores, classes = filter_yolo_boxes(boxes, scores, classes, threshold=.4)
boxes, scores, classes = non_max_suppression(boxes, scores, classes, threshold=0.9)

# draw output
draw = ImageDraw.Draw(im)
width, height = im.size
_, all_boxes, _ = np.shape(boxes)
line_width = 5

for box in range(all_boxes):

    # rescale boxes according ti size of original image
    top, left, bottom, right = boxes[:, box, :]
    top = int(top * height)
    left = int(left * width)
    bottom = int(bottom * height)
    right = int(right * width)

    # draw multiple lines to increase its width
    for line in range(line_width):
        draw.rectangle([left - 1 * line, top + 1 * line, right + 1 * line, bottom - 1 * line],
                       fill=None, outline=colors[classes[box]])

    draw.text((left, bottom), '{} {:.2f}'.format(object_list[classes[box]], scores[box]),
              font=ImageFont.truetype('arial', 16), fill=colors[classes[box]])

# TODO: specify output image path and name
im.save('output_image', "JPEG")
output_image = Image.open(os.getcwd() + '\\output_image')
output_image.show()
