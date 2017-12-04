"""
Code for extracting tiny-YOLO weights to readable form.

Input: weights and cfg
Output: 3 dicts where keys are layer numbers, values are weights

tiny-yolo weights: https://pjreddie.com/media/files/tiny-yolo-voc.weights
tiny-yolo cfg: https://github.com/pjreddie/darknet/blob/master/cfg/tiny-yolo-voc.cfg

This is a heavily modified version of:
https://github.com/allanzelener/YAD2K/blob/master/yad2k.py
"""

import os
import io
import numpy as np
import configparser
from collections import defaultdict
from keras.layers import Input
from keras import backend as K


def get_weights():

    def unique_config_sections(config_file):
        """Convert all config sections to have unique names.
        Adds unique suffixes to config sections for compability with configparser.
        """
        section_counters = defaultdict(int)
        output_stream = io.StringIO()
        with open(config_file) as fin:
            for line in fin:
                if line.startswith('['):
                    section = line.strip().strip('[]')
                    _section = section + '_' + str(section_counters[section])
                    section_counters[section] += 1
                    line = line.replace(section, str(_section))
                output_stream.write(line)
        output_stream.seek(0)
        return output_stream

    # TODO: define path to cfg and weights
    config_path = os.path.expanduser(os.getcwd() + '\\tiny-yolo.cfg')
    weights_path = os.path.expanduser(os.getcwd() + '\\tiny-yolo-voc.weights')

    # load weights
    weights_file = open(weights_path, 'rb')

    # load cfg
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    # initiate parameters
    image_height = int(cfg_parser['net_0']['height'])
    image_width = int(cfg_parser['net_0']['width'])
    prev_layer = Input(shape=(image_height, image_width, 3))

    # Darknet serializes convolutional weights as: [bias/beta, [gamma, mean, variance], conv_weights]
    prev_layer_shape = K.int_shape(prev_layer)

    # loop through layers and append weights
    layer = 0
    tiny_yolo_weights = {}
    tiny_yolo_bn_weights = {}
    tiny_yolo_conv_bias_weights = {}
    for section in cfg_parser.sections():

        print('Parsing section {}'.format(section))

        if section.startswith('convolutional'):

            # get layer parameters
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            # This assumes channel last dim_ordering
            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn' if batch_normalize else '  ', activation, weights_shape)

            # read biases
            conv_bias = np.ndarray(shape=(filters,), dtype='float32', buffer=weights_file.read(filters * 4))

            if batch_normalize:
                bn_weights = np.ndarray(shape=(3, filters), dtype='float32', buffer=weights_file.read(filters * 12))

            # read weights
            conv_weights = np.ndarray(shape=darknet_w_shape, dtype='float32', buffer=weights_file.read(weights_size * 4))

            # Caffe-style weights to Tensorflow-style:
            # (out_dim, in_dim, height, width) - > (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [conv_weights, conv_bias]

            # need to specify previous layer shape
            prev_layer_shape = [filters]

            # append
            tiny_yolo_weights[layer] = conv_weights
            tiny_yolo_bn_weights[layer] = bn_weights
            tiny_yolo_conv_bias_weights[layer] = conv_bias
            layer += 1

    return tiny_yolo_weights, tiny_yolo_conv_bias_weights, tiny_yolo_bn_weights
