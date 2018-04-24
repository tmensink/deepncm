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
"""Runs a ResNet model on the CIFAR-10 dataset."""

# Change to include as well Cifar100
#https://github.com/tensorflow/models/blob/master/research/resnet/cifar_input.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append("/home/tmensink/research/git/ncm/tf-models/models/")

import tensorflow as tf  # pylint: disable=g-bad-import-order

import resnet_ncm as resnet
import resnet_deepncm_run_loop as rrl

ALLOW_MULTIPLE_MODELS = True
DS = None

def set_dataset(dataset="cifar10"):
    print("set_dataset")
    s = type('', (), {})()

    s.NUM_IMAGES = {
        'train': 50000,
        'validation': 10000,
    }
    s.HEIGHT = 32
    s.WIDTH = 32
    s.NUM_CHANNELS = 3
    s.DEFAULT_IMAGE_BYTES = s.HEIGHT * s.WIDTH * s.NUM_CHANNELS
    if dataset == "cifar10":
        s.DATASET = 'cifar10'
        s.NUM_CLASSES = 10
        s.NUM_DATA_FILES = 5
        s.DEFAULT_MODEL_DIR = "/tmp/deepncm/cifar10_resnet/"
        s.DATA_DIR = '/tmp/deepncm/cifar10_data'
        s.DATA_PATH = 'cifar-10-batches-bin'
        s.DATA_LABEL_O = 0
        s.DATA_LABEL_B = 1
    else:
        s.DATASET = 'cifar100'
        s.DATA_LABEL_O = 1
        s.DATA_LABEL_B = 1
        s.NUM_CLASSES = 100
        s.NUM_DATA_FILES = 1
        s.DATA_DIR = '/tmp/deepncm/cifar100_data'
        s.DEFAULT_MODEL_DIR = "/tmp/deepncm/cifar100_resnet/"
        s.DATA_PATH = 'cifar-100-binary'

    s.RECORD_BYTES = s.DEFAULT_IMAGE_BYTES + s.DATA_LABEL_O + s.DATA_LABEL_B
    return s

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
    """Returns a list of filenames."""
    data_dir = os.path.join(data_dir, DS.DATA_PATH)

    assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10/CIFAR-100 data.')

    if DS.DATASET == 'cifar10':
        if is_training:
            return [
                os.path.join(data_dir, 'data_batch_%d.bin' % i)
            for i in range(1, DS.NUM_DATA_FILES + 1)
            ]
        else:
            return [os.path.join(data_dir, 'test_batch.bin')]

    else:
        if is_training:
            return [os.path.join(data_dir, 'train.bin')]
        else:
            return [os.path.join(data_dir, 'test.bin')]



def parse_record(raw_record, is_training):
    """Parse CIFAR-10/100 image and label from a raw record."""
    # Convert bytes to a vector of uint8 that is record_bytes long.
    record_vector = tf.decode_raw(raw_record, tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32
    # and then to one-hot.
    label = tf.cast(tf.slice(record_vector, [DS.DATA_LABEL_O], [DS.DATA_LABEL_B]), tf.int32)
    #label = tf.cast(record_vector[0],tf.int32)
    label = tf.one_hot(tf.squeeze(label), DS.NUM_CLASSES)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_vector, [DS.DATA_LABEL_O + DS.DATA_LABEL_B], [DS.DEFAULT_IMAGE_BYTES]),[DS.NUM_CHANNELS, DS.HEIGHT, DS.WIDTH])

    # Convert from [depth, height, width] to [height, width, depth], and cast as
    # float32.
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    image = preprocess_image(image, is_training)

    return image, label



def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, DS.HEIGHT + 8, DS.WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [DS.HEIGHT, DS.WIDTH, DS.NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             num_parallel_calls=1, multi_gpu=False):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel.
      This can be optimized per data set but for generally homogeneous data
      sets, should be approximately the number of available CPU cores.
    multi_gpu: Whether this is run multi-GPU. Note that this is only required
      currently to handle the batch leftovers, and can be removed
      when that is handled directly by Estimator.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, DS.RECORD_BYTES)

  num_images = is_training and DS.NUM_IMAGES['train'] or DS.NUM_IMAGES['validation']

  return rrl.process_record_dataset(dataset, is_training, batch_size, DS.NUM_IMAGES['train'],
                                      parse_record, num_epochs, num_parallel_calls,
                                      examples_per_epoch=num_images, multi_gpu=multi_gpu)



def get_synth_input_fn():
    return rrl.get_synth_input_fn(DS.HEIGHT, DS.WIDTH, DS.NUM_CHANNELS, DS.NUM_CLASSES)

###############################################################################
# Running the model
###############################################################################
###############################################################################
# Running the model
###############################################################################
class Cifar10Model(resnet.NCMResModel):
    """Model class with appropriate defaults for CIFAR-10 data."""

    def __init__(self, resnet_size, data_format=None, num_classes=None,version=resnet.RESNET_DEFAULT_VERSION,ncm=resnet.NCM_DEFAULT):
        """These are the parameters that work for CIFAR-10 data.

        Args:
          resnet_size: The number of convolutional layers needed in the model.
          data_format: Either 'channels_first' or 'channels_last', specifying which
            data format to use when setting up the model.
          num_classes: The number of output classes needed from the model. This
            enables users to extend the same model to their own datasets.
          version: Integer representing which version of the ResNet network to use.
            See README for details. Valid values: [1, 2]

        Raises:
          ValueError: if invalid resnet_size is chosen
        """
        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', resnet_size)

        num_blocks = (resnet_size - 2) // 6

        super(Cifar10Model, self).__init__(resnet_size=resnet_size,bottleneck=False,num_classes=num_classes,num_filters=16,kernel_size=3,conv_stride=1,first_pool_size=None,first_pool_stride=None,second_pool_size=8,second_pool_stride=1,block_sizes=[num_blocks] * 3,block_strides=[1, 2, 2],final_size=64,version=version,data_format=data_format,ncm=ncm)


def cifar10_model_fn(features, labels, mode, params):
        """Model function for CIFAR-10."""
        features = tf.reshape(features, [-1, DS.HEIGHT, DS.WIDTH, DS.NUM_CHANNELS])

        learning_rate_fn = rrl.learning_rate_with_decay(batch_size=params['batch_size'], batch_denom=params['batch_size'],num_images=DS.NUM_IMAGES['train'], boundary_epochs=[100, 150, 200],decay_rates=[1, 0.1, 0.01, 0.001],initial_learning_scale=params['initial_learning_scale'])

        # We use a weight decay of 0.0002, which performs better
        # than the 0.0001 that was originally suggested.
        weight_decay = 2e-4

        # Empirical testing showed that including batch_normalization variables
        # in the calculation of regularized loss helped validation accuracy
        # for the CIFAR-10 dataset, perhaps because the regularization prevents
        # overfitting on the small data set. We therefore include all vars when
        # regularizing and computing loss during training.
        def loss_filter_fn(_):
            return True

        print(params)
        ncm = {'method' : params['ncmmethod'],'param'  : params['ncmparam']}

        return rrl.resnet_model_fn(features, labels, mode,
                                   Cifar10Model,resnet_size=params['resnet_size'],
                                   weight_decay=weight_decay,learning_rate_fn=learning_rate_fn,
                                   momentum=0.9,data_format=params['data_format'],
                                   version=params['version'],loss_filter_fn=loss_filter_fn,
                                   multi_gpu=params['multi_gpu'],ncm=ncm)

def main(argv):
  global DS
  parser = rrl.ResnetArgParser()
  # Set defaults that are reasonable for this model.
  parser.set_defaults(resnet_size=32,
                      train_epochs=250,
                      epochs_between_evals=1,
                      batch_size=128,
                      )

  flags = parser.parse_args(args=argv[1:])

  #if not flags.dataset == DS.DATASET:
  DS = set_dataset(flags.dataset)


  flags.model_dir = DS.DEFAULT_MODEL_DIR
  flags.model_dir += flags.ncmmethod

  if flags.ncmmethod == "decaymean":
      flags.model_dir += "_d%02d" %(flags.ncmparam*100)
  elif flags.ncmmethod == "omreset":
      flags.model_dir += "_r%04d" %(flags.ncmparam)

  flags.model_dir += "_lr%5.0e" %(flags.initial_learning_scale)

  print(flags.model_dir)
  flags.data_dir = DS.DATA_DIR
  print(flags.data_dir)

  if flags.scratch > 0 and os.path.isdir(flags.model_dir):
    print ("Clear model_directory")
    import shutil
    shutil.rmtree(flags.model_dir)
  elif flags.continu > 0:
    assert os.path.isdir(flags.model_dir), "Model dir is empty, while continue is set"
  elif flags.continu == 0 and flags.scratch == 0:
    assert not os.path.isdir(flags.model_dir), "Model dir is not empty, nor continu or scratch is set"


  input_function = input_fn

  rrl.resnet_main(flags, cifar10_model_fn, input_function,shape=[DS.HEIGHT, DS.WIDTH, DS.NUM_CHANNELS])

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
