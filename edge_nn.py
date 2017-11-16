# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Edge Neural Network."""

import numpy
import tensorflow as tf
from tensorflow import flags
from tensorflow.contrib import slim

flags.DEFINE_float('learn_rate', 0.1, '')
flags.DEFINE_string('optimizer', 'pd',
                    'Training algorithm for the EdgeNN parameters. Choices are '
                    '"pd" and "adam", respecitvely, for "PercentDelta" and '
                    '"AdamOptimizer". Nonetheless, the embeddings are always '
                    'trained with PercentDelta, as per deep_edge_trainer.py')
FLAGS = flags.FLAGS

class EdgeNN(object):
  """Neural Network for representing an edge using node embeddings.

  The network maps two embedding vectors, `left` and `right`, both of size D,
  onto a scalar, indicating the edge score between the `left` and `right`
  embedding vectors.

  The score gets computed as:
      output = DNN(left) x L x R x DNN(right),

  where `DNN` is a feed-forward neural network (with trainable parameters) and
  (L, R) are two "projection" matrices.

  This class does *not* keep track of the embedding matrix. It assumes that
  embeddings are tracked outside and are provided as input during training or
  inference.
  """

  def __init__(self):
    """Sets public members to None, which are re-set in build_net()."""
    self.embeddings_a = None  # Placeholder for left embeddings.
    self.embeddings_b = None  # Placeholder for right embeddings.

    # Placeholder for batch size.
    self.batch_size = tf.placeholder(tf.int32, shape=(), name='sizes')

    ### Placeholders For training.
    self.learn_rate = tf.placeholder(tf.float32, ())

    # User is expected to feed `labels` as binary array of size `batch_size`.
    # If labels[i] == 0.0, then nodes with embeddings at embeddings_a[i] and
    # embeddings_b[i] are expected to be "negatives", but if == 1.0, they are
    # expected to be positives (i.e. direct neighbors or close and sampled
    # via Random Walks).
    self.labels = tf.placeholder(tf.float32, shape=(None,), name='labels')

    # Tensors of activations of Neural Network layers.
    self.layers = []

    # Output of the network, in terms of `embeddings_{a,b}`.
    self.output = None

    # Used for debugging.
    self.trainable_values = None

  def build_net(self, embedding_dim=None, dnn_dims=None, projection_dim=None,
                num_projections=1):
    """Creates the feed-forward DNN, projection matrices L, R, and training ops.

    Args:
      embedding_dim: Integer for D, indicating the input embedding dimension.
      dnn_dims: List of integers. Specifies the latent dimensions of hidden
        layers of DNN. Activation functions will be tf.nn.relu for all but the
        last layer will have no activation. BatchNorm will be used on all
        layers. If empty list, then no DNN will be used.
      projection_dim: Iinner dimension of the projection matrices "L" and "R".
        This is the "bottleneck" (i.e. smallest) dimension. The outer-dimension
        of "L" and "R" is inferred as last entry in `[embed_dim] + dnn_dims`.
        If set to <= 0, then no "L" nor "R" would be used. Instead, the edge
        function becomes: `w^T (DNN(left) * DNN(right))`, where * is hadamard
        product and w is a trainable vector.
    """
    if dnn_dims is None or dnn_dims == '':
      dnn_dims = []
    elif isinstance(dnn_dims, str):
      dnn_dims = map(int, dnn_dims.split(','))

    ### Placeholders For training and inference.
    # `left` and `right` embedding matrices. First (batch size) is `None` since
    # we want to support dynamic batch size. `embeddings_a` and `embeddings_b`
    # must be fed arrays of the same size, with first dimension equal to scalar
    # fed into `batch_size`.
    self.embeddings_a = tf.placeholder(
        tf.float32, shape=(None, embedding_dim), name='embeddings_a')
    self.embeddings_b = tf.placeholder(
        tf.float32, shape=(None, embedding_dim), name='embeddings_b')
    
    ### DNN.
    # Input is a concatenation of embeddings_a and embeddings_b, since they
    # both go through the same DNN transformation.
    embeddings_combined = tf.concat([
        tf.reshape(self.embeddings_a, (self.batch_size, embedding_dim)),
        tf.reshape(self.embeddings_b, (self.batch_size, embedding_dim))
    ], 0)
    self.layers.append(embeddings_combined)

    # For-loop creates the Neural Network layers. Last layer has no activation
    # but others have relu activation.
    net = embeddings_combined
    for i, f_d in enumerate(dnn_dims):
      if i < len(dnn_dims) - 1:
        net = slim.fully_connected(
            net, f_d, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(1e-6),
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': True})
      else:
        # Last layer.
        net = slim.fully_connected(
            net, f_d, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(1e-6),
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': True})
      self.layers.append(net)

    # Undo our concatenation. Set f_a to DNN(embeddings_a) and f_b to
    # DNN(embeddings_b)
    f_a = net[:self.batch_size]
    f_b = net[self.batch_size:]
    self.f_a = f_a
    self.f_b = f_b

    ### Projection with matrices "L" and "R" (i.e. g_left and g_right).
    if projection_dim > 0:
      g_outer_d = embedding_dim
      if len(dnn_dims) > 0:
        g_outer_d = dnn_dims[-1]
      self.g_lefts = []
      self.g_rights = []
      self.edge_r = []
      for i in xrange(num_projections):
        name_suffix = ''
        if i > 0:
          name_suffix = '_%i' % i
        g_left = tf.get_variable(
            name="g_left" + name_suffix, shape=(g_outer_d, projection_dim),
            regularizer=slim.regularizers.l2_regularizer(1e-6))
        g_right = tf.get_variable(
            name="g_right" + name_suffix, shape=(projection_dim, g_outer_d),
            regularizer=slim.regularizers.l2_regularizer(1e-6))

        self.g_left = g_left
        self.g_right = g_right
        self.g_lefts.append(g_left)
        self.g_rights.append(g_right)
        g_l_bottleneck = tf.matmul(f_a, g_left)
        g_r_bottleneck = tf.matmul(f_b, tf.transpose(g_right))

        self.g_l_bottleneck = g_l_bottleneck
        self.g_r_bottleneck = g_r_bottleneck
        self.layers.append(g_l_bottleneck)
        self.layers.append(g_r_bottleneck)
        output =  tf.reduce_sum(
            g_l_bottleneck * g_r_bottleneck, axis=[1])
        self.edge_r.append(tf.expand_dims(output, 1))

      if num_projections > 1:
        output = tf.concat(self.edge_r, 1)
        output = slim.batch_norm(output)
        output = tf.nn.relu(output)
        output = slim.fully_connected(output, 1, activation_fn=None)
        output = tf.squeeze(output, 1)
    else:
      output = tf.multiply(f_a, f_b)
      output = slim.fully_connected(output, 1, activation_fn=None)
      output = tf.reduce_sum(output, axis=[1])

    self.output = output
    self.build_train_op()

  def build_train_op(self):
    """Sets gradient tensors and creates tensor `train_op`."""
    self.min_objective = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=self.labels[:self.batch_size],
        logits=self.output)

    variables = [self.embeddings_a, self.embeddings_b] + tf.global_variables()
    all_losses = tf.losses.get_regularization_losses() + [self.min_objective]
    grads = tf.gradients(all_losses, variables)
    self.gradients = []
    self.gradients_for = []
    for v, g in zip(variables, grads):
      if g is None:
        continue
      self.gradients_for.append(v)
      self.gradients.append(g)

    if len(self.gradients) > 2:
      if FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(self.learn_rate)
      elif FLAGS.optimizer == 'pd':
        # Percent Delta. Works better than Adam and does not require learning
        # rate tuning.
        optimizer = tf.train.GradientDescentOptimizer(self.learn_rate)
      self.train_op = optimizer.apply_gradients(
          zip(self.gradients[2:], self.gradients_for[2:]))


  def get_gradients(self, sess, left_embeddings, right_embeddings, labels):
    """Calculates gradients w.r.t. objective.
    
    Matrices `left_embeddings` and `right_embeddings` must be of shape (b, D),
    and labels must be of shape (b), where `b` is batch size and `D` is the
    dimension of input embeddings.

    Args:
      sess: TensorFlow Session.
      left_embeddings: float32 numpy array of left embeddings, shape=(b, D).
      right_embeddings: float32 numpy array of right embeddings, shape=(b, D).
      labels: float32 numpy array of a binary vector of shape=(b). Entries must
        be 0.0 or 1.0, respectively, for negative and positive pairs at
        corresponding position in left_embeddings and right_embeddings.

    Returns:
      tuple (gradients, objective) where `gradients` contains gradients w.r.t.
      (left embeddings, right embeddings, DNN parameters, L, R). The gradients
      w.r.t. {left, right} embeddings must be applied on the embedding matrix
      by the caller, as the embedding matrix is not maintained by this class.
    """
    grads_and_objective = sess.run(
        self.gradients + [self.min_objective],
        feed_dict={
            self.embeddings_a: left_embeddings,
            self.embeddings_b: right_embeddings,
            self.batch_size: len(left_embeddings),
            self.labels: labels,
        })
    objective = grads_and_objective[-1]
    grads = grads_and_objective[:-1]

    return grads, objective

  def apply_gradients(self, sess, grads, epoch):
    """Applies `grads` to the parameters of the edge neural network.

    The optimizer is indicated using flag --optimizer. The size of grads must be
    equal to the number of tensors of the edge neural network, which must equal
    to the number of gradients returned by `get_gradients() - 2`, since the
    first two entries returned by `get_gradients()` are the gradients of
    embeddings (src, dst embeddings).

    Args:
      sess: TensorFlow session holding the parameters.
      grads: Output of get_gradients, as in `get_gradients()[0][2:]`.
      epoch: Current iteration number over train data. Used if --optimizer=pd
    
    Returns:
      The deltas in the tensors (i.e. result of the update).
    """
    if len(grads) == 0:
      return
    assert len(grads) == len(self.gradients) - 2
    if self.trainable_values is None:
      self.trainable_values = sess.run(self.gradients_for[2:])
      assert len(grads) == len(self.trainable_values)

    deltas = []
    if FLAGS.optimizer == 'pd':
      for (g, v) in zip(grads, self.trainable_values):
        mean_percent_grad  = numpy.mean(numpy.abs(g / PlusEpsilon(v)))
        deltas.append(PickLearnRate(mean_percent_grad, epoch))
    else:
      deltas = [1] * len(grads)
    feed_dict = {self.gradients[2 + i]: grads[i] * deltas[i]
                 for i in xrange(len(grads))}
    feed_dict[self.learn_rate] = FLAGS.learn_rate

    sess.run(self.train_op, feed_dict=feed_dict)
    new_trainable_values = sess.run(self.gradients_for[2:])

    deltas = [numpy.mean(abs((v0 - v1) / v1))
              for (v0, v1) in zip(self.trainable_values, new_trainable_values)]

    self.trainable_values = new_trainable_values
    return deltas


def PickLearnRate(mean_delta_grad, epoch):
  """Implementation of PercentDelta (Abu-El-Haija, 2017)."""
  epoch /= 4.0
  if epoch < 0.2:  # Less than 10% of data
    target = 0.3   # 30% change.
  elif epoch < 0.6:  # up to 30%
    target = 0.2
  elif epoch < 1.2:
    target = 0.1
  elif epoch < 2.4:
    target = 0.05
  elif epoch < 4:
    target = 0.01
  elif epoch < 8:
    target = 0.005
  elif epoch < 10:
    target = 0.001
  else:
    target = 0.0005

  # delta_grad * lr == target
  lr = target / (mean_delta_grad + 1e-5)
  return lr


def PlusEpsilon(x, eps=1e-6):
  """Element-wise add `eps` to `x` without changing sign of `x`."""
  return x + ((x < 0) * -eps) + ((x >= 0) * eps)

