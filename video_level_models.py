# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math
import numpy as np

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim


Bernoulli = tf.contrib.distributions.Bernoulli



def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(0.01))

    return {"predictions": output}


class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-5,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

# class VGG16(models.BaseModel)
#   def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
#     with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                       activation_fn=tf.nn.relu,
#                       weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
#                       weights_regularizer=slim.l2_regularizer(0.0005)):
#     net = slim.repeat(model_input, 2, slim.conv2d, 64, [3, 3], scope='conv1')
#     net = slim.max_pool2d(net, [2, 2], scope='pool1')
#     net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#     net = slim.max_pool2d(net, [2, 2], scope='pool2')
#     net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
#     net = slim.max_pool2d(net, [2, 2], scope='pool3')
#     net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
#     net = slim.max_pool2d(net, [2, 2], scope='pool4')
#     net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#     net = slim.max_pool2d(net, [2, 2], scope='pool5')
#     net = slim.fully_connected(net, 4096, scope='fc6')
#     net = slim.dropout(net, 0.5, scope='dropout6')
#     net = slim.fully_connected(net, 4096, scope='fc7')
#     net = slim.dropout(net, 0.5, scope='dropout7')
#     net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
#     return {"predictions": net}

class CNNModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, **unused_params):
      
    
    input_layer = tf.reshape(model_input, [-1,32,32,1])
    
    
    net = slim.conv2d(input_layer, 10, [3, 3])

    
    net = slim.max_pool2d(net, [32,32], [32,32], padding="same")   

    output = slim.fully_connected(
    net, vocab_size, activation_fn=tf.nn.sigmoid,
    weights_regularizer=slim.l2_regularizer(0.01))

    return {"predictions": output}
    
    
    """
    # Input Layer
    input_layer = tf.reshape(model_input, [-1, 1,32,32])


    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=96,
      kernel_size=[7, 7],
      strides=(2, 2),
      padding="same",
      activation=tf.nn.relu)
      
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[32, 32], strides=32,padding="same")

    #pool1_flat = tf.reshape(pool1, [-1, 1,1,8*96])

    #dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)
    
    
    #logits = tf.layers.dense(inputs=dense, units=vocab_size)
    
    output = slim.fully_connected(
        pool1, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(0.01))
     
    """
    
    """
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=[5, 5],
      strides=(2, 2),
      padding="same",
      activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    
    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

    pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool3_flat, units=4096, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(inputs=dropout, units=2048, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout2, units=10)
    """

class MLPModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, **unused_params):
      
    net = slim.fully_connected(model_input, 128)

    output = slim.fully_connected(
    net, vocab_size, activation_fn=tf.nn.sigmoid,
    weights_regularizer=slim.l2_regularizer(0.01))

    return {"predictions": output}

""""
This implements the Restricted Boltzmann Machine.
"""


class RBMModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, **unused_params):
        input_layer = slim.fully_connected(model_input, 1024)
        output = slim.fully_connected(
            input_layer, vocab_size, activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(0.01))
        return {"predictions": output}


class VAEModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, **unused_params):
        K = 30
        N = 10
        x = tf.placeholder(tf.float32, [vocab_size, vocab_size])
        net = slim.stack(x, slim.fully_connected, [512, 256])
        logits_y = tf.reshape(slim.fully_connected(net, K * N, activation_fn=None), [-1, K])
        q_y = tf.nn.softmax(logits_y)
        log_q_y = tf.log(q_y + 1e-20)

        tau = tf.Variable(5.0, name="temperature")
        y = tf.reshape(gumbel_softmax(logits_y, tau, hard=False), [-1, N, K])
        output = slim.stack(slim.flatten(y), slim.fully_connected, [256, 512])

        return {"predictions": output}