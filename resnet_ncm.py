# Copyright 2018 Thomas Mensink, University of Amsterdam, thomas.mensink@uva.nl
#
# Beloning to the DeepNCM repository
# DeepNCM is proposed in
#    Samantha Guerriero, Barbara Caputo, and Thomas Mensink
#    DeepNCM: Deep Nearest Class Mean Classifiers
#    ICLR Workshop 2018
#    https://openreview.net/forum?id=rkPLZ4JPM
#
# This file (resnet_ncm) has the code for different DeepNCM variantsis
# including:
#       softmax (as baseline)
#       online means (onlinemean)
#       mean condensation (omreset)
#       decay mean (decaymean)
#
"""Contains definitions for DeepNCM Residual Networks.

DeepNCM is proposed in:
[1] Samantha Guerriero, Barbara Caputo, and Thomas Mensink
    DeepNCM: Deep Nearest Class Mean Classifiers
    ICLR Workshop 2018
    https://openreview.net/forum?id=rkPLZ4JPM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import resnet_deepx as rn

RESNET_DEFAULT_VERSION = rn.DEFAULT_VERSION
NCM_DEFAULT_METHOD = "omreset"
NCM_DEFAULT_PARAMETER = 100
NCM_DEFAULT = {
    'method' : NCM_DEFAULT_METHOD,
    'param'  : NCM_DEFAULT_PARAMETER,
}

##############################################################
### Code to compute batch counts and means
##############################################################
def ncm_batch_counts(batch_x,batch_y,oneHot=True):
    if oneHot:
        by = tf.identity(batch_y)
    else:
        by = tf.one_hot(batch_y,depth=_TRAININGCLASSES,dtype=tf.float32)

    lBMn = tf.reduce_sum(by,axis=0,keepdims=True)
    lBM  = tf.matmul(by,batch_x,transpose_a=True)
    lBM  = tf.transpose(lBM)
    return lBM, lBMn

def ncm_batch_means(batch_x,batch_y,oneHot=True):
    lBC, lBMn = ncm_batch_counts(batch_x,batch_y,oneHot=oneHot)
    lBMz = lBMn + tf.cast(tf.equal(lBMn,0),dtype=tf.float32)
    lBM  = tf.transpose(lBC/lBMz)
    return lBM, lBMn

def ncm_sq_dist_bt_norm(a,b):
    anorm = tf.reshape(tf.reduce_sum(tf.square(a), 1),[-1, 1])
    bnorm = tf.reshape(tf.reduce_sum(tf.square(b), 0),[1, -1])
    d     = -2*tf.matmul(a,b,transpose_b=False)+anorm + bnorm
    return d, anorm

def ncm_sq_dist_bt(a,b):
    d, bnorm = ncm_sq_dist_bt_norm(a,b)
    return d

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

def save_batch_mean(batch_mean,batch_counts,decay_mean):
    condition = tf.cast(math_ops.greater(batch_counts,0),dtype=tf.float32)
    sbm1 = tf.multiply(batch_mean,condition)
    sbm2 = tf.multiply(tf.transpose(decay_mean),1-condition)
    return sbm1 + sbm2

def _safe_div(numerator, denominator, name):
    """Divides two tensors element-wise, returning 0 if the denominator is <= 0.
    Args:
        numerator: A real `Tensor`.
        denominator: A real `Tensor`, with dtype matching `numerator`.
        name: Name for the returned op.
    Returns:
        0 if `denominator` <= 0, else `numerator` / `denominator`

    Copied from TensorFlow Metrics
    """
    t = math_ops.truediv(numerator, denominator)
    zero = array_ops.zeros_like(t, dtype=denominator.dtype)
    condition = math_ops.greater(denominator, zero)
    zero = math_ops.cast(zero, t.dtype)
    return array_ops.where(condition, t, zero, name=name)


class NCMResModel(rn.ResNetX):
  """NCM ResNet class for building the DeepNCM Resnet Model."""

  def __init__(self, resnet_size, bottleneck, num_classes, num_filters,
               kernel_size,
               conv_stride, first_pool_size, first_pool_stride,
               second_pool_size, second_pool_stride, block_sizes, block_strides,
               final_size, version=RESNET_DEFAULT_VERSION, ncm=NCM_DEFAULT, data_format=None):
    super(NCMResModel,self).__init__(resnet_size, bottleneck, num_classes, num_filters,
                 kernel_size,
                 conv_stride, first_pool_size, first_pool_stride,
                 second_pool_size, second_pool_stride, block_sizes, block_strides,
                 final_size, version, data_format)
    self.ncmmethod = ncm['method'].casefold()
    self.ncmparam  = ncm['param']

    if self.ncmmethod == "decaymean":
        assert 0 <= self.ncmparam < 1, "Decay means requires ncmparam between 0 and 1"

    self.iter  = tf.get_variable("iter", [],dtype=tf.float32,trainable=False, initializer=tf.initializers.constant(0))
    self.total = tf.get_variable("total",[final_size,num_classes],dtype=tf.float32,trainable=False, initializer=tf.initializers.constant(0))
    self.count = tf.get_variable("count",[1,num_classes],dtype=tf.float32,trainable=False, initializer=tf.initializers.constant(0))


  def get_mean_and_batch_mean(self,deep_x=None,labels=None):
      bmean,bcounts = ncm_batch_means(deep_x,labels)
      return _safe_div(self.total,self.count,name="deepmean"), bmean, bcounts

  def get_relative_mean_distance(self,deep_x=None,labels=None):
      bmean,bcounts = ncm_batch_means(deep_x,labels)
      dm,dmnorm = ncm_sq_dist_bt_norm(bmean,_safe_div(self.total, self.count,name='deepmean'))
      rdist = _safe_div(tf.diag_part(dm),dmnorm,name='relativedist')

      return rdist, tf.cast(tf.greater(bcounts,0),rdist.dtype)


  def get_reset_op(self,update_op):
        reset_total_op = state_ops.assign(self.total,update_op,use_locking=True)
        with ops.control_dependencies([update_op]):
            reset_count_op = state_ops.assign(self.count,array_ops.ones_like(self.count),use_locking=True)

        reset_op = _safe_div(reset_total_op,reset_count_op, 'reset_op')
        return reset_op

  def get_ncm_ops(self,deep_x=None,labels=None):
    iter_op = state_ops.assign_add(self.iter,tf.ones([]))

    if self.ncmmethod == "onlinemean" or self.ncmmethod == "omreset":
        batchsums,batchcnts = ncm_batch_counts(deep_x,labels)
        update_total_op = state_ops.assign_add(self.total, batchsums,use_locking=True)
        with ops.control_dependencies([batchsums]):
            update_count_op = state_ops.assign_add(self.count, batchcnts,use_locking=True)

        update_op = _safe_div(update_total_op, update_count_op, 'update_op')

    if self.ncmmethod == "decaymean":
       batchmeans,batchcnts = ncm_batch_means(deep_x,labels)
       batchcnts = tf.transpose(batchcnts)
       sbm = save_batch_mean(batchmeans,batchcnts,self.total)
       sbm = tf.transpose(sbm)
       ndm = self.ncmparam * self.total + (1-self.ncmparam) * sbm
       update_total_op = state_ops.assign(self.total,ndm, use_locking=True)
       with ops.control_dependencies([ndm]):
           update_count_op = state_ops.assign(self.count, array_ops.ones_like(self.count),use_locking=True)

       update_op = _safe_div(update_total_op, update_count_op, 'update_op')

    if self.ncmmethod == "onlinemean" or self.ncmmethod == "decaymean":
        ncm_op = tf.group(iter_op,update_op)
    elif self.ncmmethod == "omreset":
        ncm_op = tf.cond(tf.equal(tf.mod(self.iter,self.ncmparam),0), false_fn=lambda: tf.group(iter_op,update_op),true_fn=lambda: tf.group(iter_op,self.get_reset_op(update_op)))
    else: #SOFTMAX case
        ncm_op = iter_op

    return ncm_op

  def __call__(self, inputs, training):
    deepx = super(NCMResModel,self).__call__(inputs, training)
    deepx = tf.identity(deepx, 'deep-representation')

    deepmean = _safe_div(self.total, self.count, 'deepmean')
    deepmean = tf.identity(deepmean,name="DeepMeanValue")

    if self.ncmmethod == "softmax":
        logits = tf.layers.dense(inputs=deepx, units=self.num_classes)

    elif self.ncmmethod == "onlinemean" or self.ncmmethod == "omreset" or self.ncmmethod == "decaymean":
        logits   = -ncm_sq_dist_bt(deepx,deepmean)

    logits = tf.identity(logits, 'logits')
    return logits, deepx, deepmean
