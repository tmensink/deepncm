# Copyright 2018 Thomas Mensink, University of Amsterdam, thomas.mensink@uva.nl
#
# Beloning to the DeepNCM repository
# DeepNCM is proposed in
#    Samantha Guerriero, Barbara Caputo, and Thomas Mensink
#    DeepNCM: Deep Nearest Class Mean Classifiers
#    ICLR Workshop 2018
#    https://openreview.net/forum?id=rkPLZ4JPM
#
# This file (cifar10cifar10_download_and_extract) is based on the
# TensorFlow Models Official ResNet library (release 1.8.0/1.7.0)
# https://github.com/tensorflow/models/tree/master/official/resnet
# It is changed to include both CIFAR10 as well as CIFAR100 dataset

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Downloads and extracts the binary version of the CIFAR-10/CIFAR-100 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

C10_DIR = '/tmp/deepncm/cifar10_data'
C100_DIR = '/tmp/deepncm/cifar100_data'

C10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
C100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'

parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset', type=str, default='cifar10',
    help='Dataset to download Cifar10 or Cifar100')

parser.add_argument(
    '--data_dir', type=str, default=C10_DIR,
    help='Directory to download data and extract the tarball')


def main(_):
  """Download and extract the tarball from Alex's website."""
  print(FLAGS.dataset)

  if FLAGS.dataset == 'cifar10':
      DATA_URL = C10_URL
  else:
      DATA_URL = C100_URL
      if FLAGS.data_dir == C10_DIR:
          FLAGS.data_dir = C100_DIR

  print(FLAGS.data_dir)
  print(DATA_URL)

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(FLAGS.data_dir, filename)

  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, 100.0 * count * block_size / total_size))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath, 'r:gz').extractall(FLAGS.data_dir)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
