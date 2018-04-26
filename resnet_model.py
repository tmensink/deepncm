# Copyright 2018 Thomas Mensink, University of Amsterdam, thomas.mensink@uva.nl
#
# Beloning to the DeepNCM repository
# DeepNCM is proposed in
#    Samantha Guerriero, Barbara Caputo, and Thomas Mensink
#    DeepNCM: Deep Nearest Class Mean Classifiers
#    ICLR Workshop 2018
#    https://openreview.net/forum?id=rkPLZ4JPM
#
# This file (resnet_model) is based on resnet_model from the
# TensorFlow Models Official ResNet library (release 1.8.0/1.7.0)
# https://github.com/tensorflow/models/tree/master/official/resnet
#
# It contains the ResNet code to mimic th resnet_model making use of resnet_deepx.
# ==============================================================================
"""Contains definitions for Residual Networks based on resnet_deepx
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import resnet_deepx as rn

DEFAULT_VERSION = rn.DEFAULT_VERSION

class ResNetModel(rn.ResNetX):
  """Base class for building the Resnet Model."""

  def __init__(self, resnet_size, bottleneck, num_classes, num_filters,
               kernel_size,
               conv_stride, first_pool_size, first_pool_stride,
               second_pool_size, second_pool_stride, block_sizes, block_strides,
               final_size, version=DEFAULT_VERSION, data_format=None):
    super(ResNetModel,self).__init__(resnet_size, bottleneck, num_classes, num_filters,
                 kernel_size,
                 conv_stride, first_pool_size, first_pool_stride,
                 second_pool_size, second_pool_stride, block_sizes, block_strides,
                 final_size, version, data_format)

  def __call__(self, inputs, training):
    inputs = super(ResNetModel,self).__call__(inputs, training)
    inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
    inputs = tf.identity(inputs, 'final_dense')

    return inputs
