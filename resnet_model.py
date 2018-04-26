# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
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
