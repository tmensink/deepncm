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
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf  # pylint: disable=g-bad-import-order

import numpy as np
#np.random.seed(234)

import resnet_ncm as rncm
import resnet_run_loop as rrl
ALLOW_MULTIPLE_MODELS = True

from official.utils.arg_parsers import parsers
from official.utils.export import export
from official.utils.logging import hooks_helper
from official.utils.logging import logger


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_parallel_calls=1,
                           examples_per_epoch=0, multi_gpu=False):
    return rrl.process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                parse_record_fn, num_epochs=num_epochs, num_parallel_calls=num_parallel_calls,
                examples_per_epoch=examples_per_epoch, multi_gpu=multi_gpu)

def get_synth_input_fn(height, width, num_channels, num_classes):
    return rrl.get_synth_input_fn(height, width, num_channles, num_classes)

################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates,initial_learning_scale=0.1):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.

  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = initial_learning_scale * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries, vals)

  return learning_rate_fn


def variable_summary(var,namescope="Variable_Summary"):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(namescope):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, version, loss_filter_fn=None, multi_gpu=False, ncm=rncm.NCM_DEFAULT):
  """Shared functionality for different resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    version: Integer representing which version of the ResNet network to use.
      See README for details. Valid values: [1, 2]
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    multi_gpu: If True, wrap the optimizer in a TowerOptimizer suitable for
      data-parallel distribution across multiple GPUs.

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """
  # Generate a summary node for the images
  tf.summary.image('images', features, max_outputs=6)

  model = model_class(resnet_size, data_format=data_format, num_classes=labels.shape[1].value, version=version,ncm=ncm)

  #tf.train.init_from_checkpoint('/tmp/deepncm/cifar10_resnet/onlinemean_lr1e-01/model.ckpt-1174',{'/','/'})

  logits, deep_x, deepmean = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
  }


  dm = tf.identity(deepmean,"DM")
  #variable_summary(model.m_deep,"DeepMean")
  #variable_summary(model.m_sum,"M-Sum")
  #variable_summary(model.m_cnt,"M-Count")
  #tf.summary.scalar("NCM-iter",model.m_iter)
  if not (model.ncmmethod == "softmax"):
      rdist,mmsk = model.get_relative_mean_distance(deep_x=deep_x,labels=labels)
      mcmd = tf.metrics.mean(rdist,weights=mmsk)
      rmd = tf.identity(mcmd[1],name="rmd")
      rmd = tf.summary.scalar('rmd', rmd)
      predictions['rmd'] = tf.identity(rmd,name="rmd")

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        }
        )

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name
  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if loss_filter_fn(v.name)])
  tf.summary.scalar('l2_loss', l2_loss)
  loss = cross_entropy + l2_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    ncm_ops = model.get_ncm_ops(deep_x=deep_x,labels=labels)

    # Create a tensor named learning_rate for logging purposes
    global_step = tf.train.get_or_create_global_step()
    learning_rate = learning_rate_fn(global_step)
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    # Create loss_op using Gradient clipping
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum)
    gavs = optimizer.compute_gradients(loss)
    gavsc= [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gavs]
    loss_op = optimizer.apply_gradients(gavsc,global_step=global_step)

    # Update ops from Graph
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    train_op = tf.group(loss_op, update_ops, ncm_ops)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])

  dm,bm,bmc = model.get_mean_and_batch_mean(deep_x=deep_x,labels=labels)

  metrics = {'accuracy': accuracy}
  if not (model.ncmmethod == "softmax"):
      metrics['mcmdistance'] = mcmd
      metrics['batchmeans'] = tf.metrics.mean_tensor(tf.transpose(bm),weights=bmc)
      metrics['deepmean'] = tf.metrics.mean_tensor(dm)

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def resnet_main(flags, model_function, input_function, shape=None):
  """Shared main loop for ResNet Models.

  Args:
    flags: FLAGS object that contains the params for running. See
      ResnetArgParser for created flags.
    model_function: the function that instantiates the Model and builds the
      ops for train/eval. This will be passed directly into the estimator.
    input_function: the function that processes the dataset and returns a
      dataset that the estimator can train on. This will be wrapped with
      all the relevant flags for running and passed to estimator.
    shape: list of ints representing the shape of the images used for training.
      This is only used if flags.export_dir is passed.
  """

  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Create session config based on values of inter_op_parallelism_threads and
  # intra_op_parallelism_threads. Note that we default to having
  # allow_soft_placement = True, which is required for multi-GPU and not
  # harmful for other modes.
  session_config = tf.ConfigProto(
      inter_op_parallelism_threads=flags.inter_op_parallelism_threads,
      intra_op_parallelism_threads=flags.intra_op_parallelism_threads,
      allow_soft_placement=True)

  if ALLOW_MULTIPLE_MODELS:
      session_config.gpu_options.allow_growth = True

  # Set up a RunConfig to save checkpoint and set session config.
  run_config = tf.estimator.RunConfig().replace(
     save_checkpoints_secs = 5*60,  # Save checkpoints every X minutes.
     keep_checkpoint_max = 1000,    # Retain the 1000 most recent checkpoints.
     #tf_random_seed = 5739,         # Set random seed for "reproducible" results
     save_summary_steps = 10,       # Number of steps between summaries
     session_config=session_config)

  classifier = tf.estimator.Estimator(
      model_fn=model_function, model_dir=flags.model_dir, config=run_config,
      params={
          'resnet_size': flags.resnet_size,
          'data_format': flags.data_format,
          'batch_size': flags.batch_size,
          'multi_gpu': flags.multi_gpu,
          'version': flags.version,
          'ncmmethod': flags.ncmmethod,
          'ncmparam' : flags.ncmparam,
          'initial_learning_scale' : flags.initial_learning_scale
      })

  if flags.benchmark_log_dir is not None:
    benchmark_logger = logger.BenchmarkLogger(flags.benchmark_log_dir)
    benchmark_logger.log_run_info("resnet")
  else:
    benchmark_logger = None

  for _ in range(flags.train_epochs // flags.epochs_between_evals):
    train_hooks = hooks_helper.get_train_hooks(
        flags.hooks,
        batch_size=flags.batch_size,
        benchmark_log_dir=flags.benchmark_log_dir)
    #tensors_to_log = {"iter": "m_iter","deep-cnt": "m_cnt", "deep-sum": "m_sum"}
    #logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)


    print('Starting a training cycle.')

    def input_fn_train():
      return input_function(True, flags.data_dir, flags.batch_size,
                            flags.epochs_between_evals,
                            flags.num_parallel_calls, flags.multi_gpu)

    classifier.train(input_fn=input_fn_train, hooks=train_hooks,max_steps=flags.max_train_steps)

    print('Starting to evaluate.')
    # Evaluate the model and print results
    def input_fn_eval():
      return input_function(False, flags.data_dir, flags.batch_size,
                            1, flags.num_parallel_calls, flags.multi_gpu)

    # flags.max_train_steps is generally associated with testing and profiling.
    # As a result it is frequently called with synthetic data, which will
    # iterate forever. Passing steps=flags.max_train_steps allows the eval
    # (which is generally unimportant in those circumstances) to terminate.
    # Note that eval will run for max_train_steps each loop, regardless of the
    # global_step count.
    eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                       steps=flags.max_train_steps)
    print(eval_results)

    if benchmark_logger:
      benchmark_logger.log_estimator_evaluation_result(eval_results)

    if flags.export_dir is not None:
        # Exports a saved model for the given classifier.
        input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
            shape, batch_size=flags.batch_size)
        classifier.export_savedmodel(flags.export_dir, input_receiver_fn)


class ResnetArgParser(argparse.ArgumentParser):
  """Arguments for configuring and running a Resnet Model."""

  def __init__(self, resnet_size_choices=None):
    super(ResnetArgParser, self).__init__(parents=[
        parsers.BaseParser(),
        parsers.PerformanceParser(),
        parsers.ImageModelParser(),
        parsers.ExportParser(),
        parsers.BenchmarkParser(),
    ])

    self.add_argument('--dataset','-d',default="cifar10",
        help='Which dataset to use (currently cifar10/cifar100)'
    )

    self.add_argument(
        '--version', '-v', type=int, choices=[1, 2],
        default=rncm.RESNET_DEFAULT_VERSION,
        help='Version of ResNet. (1 or 2) See README.md for details.'
    )

    self.add_argument(
        '--resnet_size', '-rs', type=int, default=50,
        choices=resnet_size_choices,
        help='[default: %(default)s] The size of the ResNet model to use.',
        metavar='<RS>' if resnet_size_choices is None else None
    )

    self.add_argument(
        '--continu',type=int,default=0,
        help='Continue with an existing model, or start from scratch'
    )

    self.add_argument(
        '--scratch',type=int,default=0,
        help='Start from scratch even if model exist'
    )

    self.add_argument(
        '--ncmmethod', default=rncm.NCM_DEFAULT_METHOD,
        help='[default: %(default)s] Which NCM method to use',
    )

    self.add_argument(
        '--ncmparam', default=rncm.NCM_DEFAULT_PARAMETER, type=float,
        help='[default: %(default)s] additional NCM parameter to use',
    )

    self.add_argument(
        '--initial_learning_scale', '-l', default=0.1, type=float,
        help='Intial Learning Scale (default: %(default)s)',
    )
