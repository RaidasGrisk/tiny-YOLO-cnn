"""
Code for building tiny-yolo.

Model architecture cfg:
https://github.com/pjreddie/darknet/blob/master/cfg/tiny-yolo-voc.cfg

To make things clearer, the architecture is specified in this file.
This specification matches .cfg specs.

Useful resource:
https://github.com/allanzelener/YAD2K/
"""

from get_weights import get_weights
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model


def model():

    # architecture
    parameters = {0: {'filters': 16,
                      'size': 3,
                      'stride': 1,
                      'maxpool_size': 2,
                      'maxpool_stride': 2,
                      'batch_normalize': True,
                      'activation': 'leakyRelu'
                      },
                  1: {'filters': 32,
                      'size': 3,
                      'stride': 1,
                      'maxpool_size': 2,
                      'maxpool_stride': 2,
                      'batch_normalize': True,
                      'activation': 'leakyRelu'
                      },
                  2: {'filters': 64,
                      'size': 3,
                      'stride': 1,
                      'maxpool_size': 2,
                      'maxpool_stride': 2,
                      'batch_normalize': True,
                      'activation': 'leakyRelu'
                      },
                  3: {'filters': 128,
                      'size': 3,
                      'stride': 1,
                      'maxpool_size': 2,
                      'maxpool_stride': 2,
                      'batch_normalize': True,
                      'activation': 'leakyRelu'
                      },
                  4: {'filters': 256,
                      'size': 3,
                      'stride': 1,
                      'maxpool_size': 2,
                      'maxpool_stride': 2,
                      'batch_normalize': True,
                      'activation': 'leakyRelu'
                      },
                  5: {'filters': 512,
                      'size': 3,
                      'stride': 1,
                      'maxpool_size': 2,
                      'maxpool_stride': 1,
                      'batch_normalize': True,
                      'activation': 'leakyRelu'
                      },
                  6: {'filters': 1024,
                      'size': 3,
                      'stride': 1,
                      'maxpool_size': False,
                      'maxpool_stride': False,
                      'batch_normalize': True,
                      'activation': 'leakyRelu'
                      },
                  7: {'filters': 512,
                      'size': 3,
                      'stride': 1,
                      'maxpool_size': False,
                      'maxpool_stride': False,
                      'batch_normalize': True,
                      'activation': 'leakyRelu'
                      },
                  8: {'filters': 425,
                      'size': 1,
                      'stride': 1,
                      'pad': 'same',
                      'maxpool_size': False,
                      'maxpool_stride': False,
                      'batch_normalize': False,
                      'activation': 'linear'
                      }
                  }

    # init model variables
    weights, bias_weights, bn_weights = get_weights()
    layer_values = Input(shape=(416, 416, 3))

    # model
    stacked_layers = [layer_values]
    for layer, params in parameters.items():

        # conv
        layer_values = Conv2D(
            filters=params['filters'],
            kernel_size=params['size'],
            strides=params['stride'],
            padding='same',
            weights=weights[layer],
            use_bias=not params['batch_normalize'],  # do not use if batch norm
            kernel_regularizer=l2(0.0005))(layer_values)  # decay specified in cfg

        # normalization
        if params['batch_normalize']:
            layer_values = BatchNormalization(
                weights=[bn_weights[layer][0],  # scale gamma
                         bias_weights[layer],  # shift beta
                         bn_weights[layer][1],  # running var
                         bn_weights[layer][2]  # running mean
                         ])(layer_values)

        # activation
        if params['activation'] == 'leakyRelu':
            layer_values = LeakyReLU(alpha=0.1)(layer_values)
            stacked_layers.append(layer_values)
        else:
            stacked_layers.append(layer_values)

        # pooling
        if params['maxpool_size']:
            layer_values = MaxPooling2D(
                pool_size=params['maxpool_size'],
                strides=params['maxpool_stride'],
                padding='same')(layer_values)
            stacked_layers.append(layer_values)

    return Model(inputs=stacked_layers[0], outputs=stacked_layers[-1])
