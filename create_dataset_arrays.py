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
"""Creates dataset files to be used by `deep_edge_trainer.py`.

The input should be edge-list text file, with lines like "node1 node2", where
the nodes can be strings or ints. The line depicts a relationship node1-node2
(undirected) or node1->node2 (directed). The node IDs (e.g. "node1") will be
mapped to integers in [0, |V| - 1], where |V| is the number of graph nodes. The
mapping will be saved in `index.pkl`.

By default, the input graph (edge-list) will be partitioned into train and test,
both of equal number of edges, where the train partition is connected (following
node2vec). 

The output directory will be populated with files:
  train.txt.npy: int32.numpy array (|E|/2, 2) containing training edges.
  test.txt.npy: int32.numpy array (|E|/2, 2) containing test edges.
  train.neg.txt.npy: int32.numpy array (|E|/2, 2) containing negative trai
    edges, sampled from compliment of (train.txt.npy).
  test.neg.txt.npy: int32.numpy array (|E|/2, 2) containing negative test edges,
    sampled from compliment of (test.txt.npy union train.txt.npy)
  train.pairs.<i>.txt.npy: One or more training pair numpy arrays [size (?, 2)].

See doc of `CreateDatasetFiles()` for complete list of files and description.

To run, you should download node2vec.py from
https://github.com/aditya-grover/node2vec/blob/master/src/node2vec.py
and place in the same directory as this file. If you do not download it, it will
automatically be downloaded on your behalf.
"""

import cPickle
import copy
import random
import networkx as nx
import numpy
import os
import sys

import tensorflow as tf

from tensorflow import flags

from third_party.node2vec import node2vec


flags.DEFINE_string('input', '',
                    'Path to edge-list textfile. Required unless '
                    '--only_simulate_walks is set.')
flags.DEFINE_boolean('only_simulate_walks', False,
                     'If set, train.txt.npy will be read from --output_dir, '
                     'random walks will be simulated, out their output will be '
                     'written to --output_dir')
flags.DEFINE_string('output_dir', '',
                    'Directory where training files will be written.')
flags.DEFINE_boolean('directed', False, 'Must be set if graph is directed.')
flags.DEFINE_boolean('partition', True,
                     'If set (default), separates a test split, containing '
                     'half of the edges. In which case, train graph will be '
                     'connected.')

flags.DEFINE_integer('num_walks', 5, 'Number of walks per node.')
flags.DEFINE_integer('walk_length', 40,
                     'Length of each walk. Total number of pairs will be '
                     'O(walk_length * num_walks * num nodes * context^2)')
flags.DEFINE_integer('context', 3,
                     'Size of context from each side (right and left). If '
                     '--directed, then context is only taken from right side')

FLAGS = flags.FLAGS

# node2vec parameters
N2V_P=1.0
N2V_Q=1.0


def LargestSubgraph(graph):
  """Returns the Largest connected-component of `graph`."""
  if graph.__class__ == nx.Graph:
    return LargestUndirectedSubgraph(graph)
  elif graph.__class__ == nx.DiGraph:
    largest_undirected_cc = LargestUndirectedSubgraph(nx.Graph(graph))
    directed_subgraph = nx.DiGraph()
    for (n1, n2) in graph.edges():
      if n2 in largest_undirected_cc and n1 in largest_undirected_cc[n2]:
        directed_subgraph.add_edge(n1, n2)

    return directed_subgraph

def LargestUndirectedSubgraph(graph):
  """Returns the largest connected-component of undirected `graph`."""
  if nx.is_connected(graph):
    return graph

  cc = list(nx.connected_component_subgraphs(graph))
  sizes = map(len, cc)
  sizes_and_cc = zip(sizes, cc)
  sizes_and_cc.sort()

  return sizes_and_cc[-1][1]

def SampleTestEdgesAndPruneGraph(graph, remove_percent=0.5, check_every=5):
  """Removes and returns `remove_percent` of edges from graph.

  Removal is random but makes sure graph stays connected."""
  graph = copy.deepcopy(graph)
  undirected_graph = graph.to_undirected()

  edges = copy.deepcopy(graph.edges())
  random.shuffle(edges)
  remove_edges = int(len(edges) * remove_percent)
  num_edges_removed = 0
  currently_removing_edges = []
  removed_edges = []
  last_printed_prune_percentage = -1
  for j in xrange(len(edges)):
    n1, n2 = edges[j]
    graph.remove_edge(n1, n2)
    if n1 not in graph[n2]:
      undirected_graph.remove_edge(*(edges[j]))
    currently_removing_edges.append(edges[j])
    if j % check_every == 0:
      if nx.is_connected(undirected_graph):
        num_edges_removed += check_every
        removed_edges += currently_removing_edges
        currently_removing_edges = []
      else:
        for i in xrange(check_every):
          graph.add_edge(*(edges[j - i]))
          undirected_graph.add_edge(*(edges[j - i]))
        currently_removing_edges = []
        if not nx.is_connected(undirected_graph):
          print '  DID NOT RECOVER :('
          return None
    prunned_percentage = int(100 * len(removed_edges) / remove_edges)
    rounded = (prunned_percentage / 10) * 10
    if rounded != last_printed_prune_percentage:
      last_printed_prune_percentage = rounded
      print 'Partitioning into train/test. Progress=%i%%' % rounded

    if len(removed_edges) >= remove_edges:
      break

  return graph, removed_edges


def SampleNegativeEdges(graph, num_edges):
  """Samples `num_edges` edges from compliment of `graph`."""
  random_negatives = set()
  nodes = list(graph.nodes())
  while len(random_negatives) < num_edges:
    i1 = random.randint(0, len(nodes) - 1)
    i2 = random.randint(0, len(nodes) - 1)
    if i1 == i2:
      continue
    if i1 > i2:
      i1, i2 = i2, i1
    n1 = nodes[i1]
    n2 = nodes[i2]
    if graph.has_edge(n1, n2):
      continue
    random_negatives.add((n1, n2))

  return random_negatives


def RandomNegativesPerNode(graph, negatives_per_node=400):
  """For every node u in graph, samples 20 (u, v) where v is not in graph[u]."""
  negatives = []
  node_list = list(graph.nodes())
  num_nodes = len(node_list)
  print_every = num_nodes / 10
  for i, n in enumerate(node_list):
    found_negatives = 0
    if i % print_every == 0:
      print 'Finished sampling negatives for %i / %i nodes' % (i, num_nodes)
    while found_negatives < negatives_per_node:
      n2 = node_list[random.randint(0, num_nodes - 1)]
      if n == n2 or n2 in graph[n]:
        continue
      negatives.append((n, n2))
      found_negatives += 1
  return negatives


def NumberNodes(graph):
  """Returns a copy of `graph` where nodes are replaced by incremental ints."""
  node_list = sorted(graph.nodes())
  index = {n: i for (i, n) in enumerate(node_list)}

  newgraph = graph.__class__()
  for (n1, n2) in graph.edges():
    newgraph.add_edge(index[n1], index[n2])

  return newgraph, index


class WalkPairsWriter(object):
  """Writes one or more int numpy.array of size (S, 2).
  
  Where `S` is the size of the array, up to `self.capacity`. The total number
  of pairs should be the number of times `AddPair` is called.
  """

  def __init__(self, file_format):
    """file_format must contain %i."""
    self.file_format = file_format
    self.capacity = 1000000   # 1 million.
    self.pairs = []
    self.next_file_id = 0

  def AddPair(self, n1, n2):
    self.pairs.append((n1, n2))
    if len(self.pairs) > self.capacity:
      self.Write()

  def Write(self):
    if len(self.pairs) == 0:
      return
    file_name = self.file_format % self.next_file_id
    random.shuffle(self.pairs)
    pairs_arr = numpy.array(self.pairs, dtype='int32')
    numpy.save(file_name, pairs_arr)
    self.pairs = []
    self.next_file_id += 1


def MakeDirectedNegatives(positive_edges):
  positive_set = set([(u, v) for (u, v) in list(positive_edges)])
  directed_negatives = []
  for (u, v) in positive_set:
    if (v, u) not in positive_set:
      directed_negatives.append((v, u))
  return numpy.array(directed_negatives, dtype='int32')


def CreateDatasetFiles(graph, output_dir, partition=True):
  """Writes a number of dataset files to `output_dir`.

  Args:
    graph: nx.Graph or nx.DiGraph to simulate walks on and extract negatives.
    output_dir: files will be written in this directory, including:
      {train, train.neg, test, test.neg}.txt.npy, index.pkl, and
      if flag --directed is set, test.directed.neg.txt.npy.
      The files {train, train.neg}.txt.npy are used for model selection;
      {test, test.neg, test.directed.neg}.txt.npy will be used for calculating
      eval metrics; index.pkl contains information about the graph (# of nodes,
      mapping from original graph IDs to new assigned integer ones in
      [0, largest_cc_size-1].
    partition: If set largest connected component will be used and data will 
      separated into train/test splits.

  Returns:
    The training graph, after node renumbering.
  """
  num_floats = num_walks * walk_length * len(graph)
  num_floats *= (context_left + context_right) ** 2
  print "Writing up to %i training pairs, with size = %0.1f megabytes." % (
      num_floats, (num_floats * 4)/1000000.0)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  original_size = len(graph)
  if partition:
    graph = LargestSubgraph(graph)
    size_largest_cc = len(graph)
  else:
    size_largest_cc = -1
  graph, index = NumberNodes(graph)

  if partition:
    train_graph, test_edges = SampleTestEdgesAndPruneGraph(graph)
  else:
    train_graph, test_edges = graph, []

  # Sample negatives, to be equal to number of `test_edges` * 2.
  random_negatives = list(
      SampleNegativeEdges(graph, len(test_edges) + len(train_graph.edges())))
  random.shuffle(random_negatives)
  test_negatives = random_negatives[:len(test_edges)]
  # These are only used for evaluation, never training.
  train_eval_negatives = random_negatives[len(test_edges):]

  test_negatives = numpy.array(test_negatives, dtype='int32')
  test_edges = numpy.array(test_edges, dtype='int32')
  train_edges = numpy.array(train_graph.edges(), dtype='int32')
  train_eval_negatives = numpy.array(train_eval_negatives, dtype='int32')

  numpy.save(os.path.join(output_dir, 'train.txt'), train_edges)
  numpy.save(os.path.join(output_dir, 'train.neg.txt'), train_eval_negatives)
  numpy.save(os.path.join(output_dir, 'test.txt'), test_edges)
  numpy.save(os.path.join(output_dir, 'test.neg.txt'), test_negatives)
  if FLAGS.directed:
    directed_negatives = MakeDirectedNegatives(
        numpy.concatenate([train_edges, test_edges], axis=0))
    directed_negatives = numpy.concatenate([directed_negatives, test_negatives],
                                           axis=0)
    numpy.save(
        os.path.join(output_dir, 'test.directed.neg.txt'), directed_negatives)

  cPickle.dump({
      'index': index,
      'original_num_nodes': original_size,
      'largest_cc_num_nodes': size_largest_cc,
      'num_pos_test_edges': len(test_edges),
      'num_neg_test_edges': len(test_negatives),
      'num_pos_train_edges': len(train_edges),
      'num_neg_train_edges': len(train_eval_negatives),
  }, open(os.path.join(output_dir, 'index.pkl'), 'w'))

  return train_graph


def SimulateWalks(train_graph, output_dir, num_walks=10, walk_length=80,
                  context_left=3, context_right=3, p=N2V_P, q=N2V_Q):
  """Simulates Random Walks on `train_graph`, writing onto `output_dir`.

  Args:
    train_graph: nx.Graph or nx.DiGraph to simulate walks on and extract
      negatives.
    output_dir: files will be written in this directory, including:
      train.neg_per_node.txt.npy and train.pairs.<i>.txt.npy, for integer <i> in
      [0, num_walks - 1]. These files will be used for training the linear
      approximation of the Graph Likelihood objective.
    num_walks: Number of walks per node.
    walk_length: Walk length from every node.
    context_left: left offset from central word, inclusive.
    context_right: right offset from central word, inclusive.
    p: Node2vec's p parameter.
    q: Node2vec's q parameter.
  """
  train_negatives_per_node = RandomNegativesPerNode(
      train_graph, negatives_per_node=400)
  train_negatives_per_node = numpy.array(train_negatives_per_node,
                                         dtype='int32')
  numpy.save(os.path.join(output_dir, 'train.neg_per_node.txt'),
             train_negatives_per_node)

  for edge in train_graph.edges():
    train_graph[edge[0]][edge[1]]['weight'] = 1
  directed = (train_graph.__class__ == nx.DiGraph)
  node2vec_graph = node2vec.Graph(train_graph, is_directed=directed, p=p, q=q)
  node2vec_graph.preprocess_transition_probs()

  pairs_writer = WalkPairsWriter(os.path.join(output_dir, 'train.pairs.%i'))
  for unused_j in xrange(FLAGS.num_walks):
    walks = node2vec_graph.simulate_walks(1, FLAGS.walk_length)

    for c, node_list in enumerate(walks):
      if c % 1000 == 0:
        print 'Writing Walk Pairs %i / %i' % (c, len(walks))
      for i in xrange(len(node_list)):
        start_i = max(0, i - context_left)
        end_i = min(len(node_list), i + context_right + 1)
        for k in xrange(start_i, end_i):
          # if i == k: continue
          pairs_writer.AddPair(node_list[i], node_list[k])

  pairs_writer.Write()

  print 'All Done. Nodes = %i' % len(train_graph)

def main(unused_argv):
  if FLAGS.directed:
    graph = nx.DiGraph()
  else:
    graph = nx.Graph()

  if not FLAGS.only_simulate_walks:
    # Read graph
    graph = nx.read_edgelist(FLAGS.input, create_using=graph)

    # Create dataset files.
    graph = CreateDatasetFiles(
        graph, FLAGS.output_dir, num_walks=FLAGS.num_walks,
        context_right=FLAGS.context, context_left=FLAGS.context*FLAGS.directed,
        walk_length=FLAGS.walk_length, p=N2V_P, q=N2V_Q)
  else:
    if os.path.exists(
        os.path.join(FLAGS.output_dir, 'test.directed.neg.txt.npy')):
      graph = nx.DiGraph()
      FLAGS.directed = True

    # Only simulating walks. Read graph from --output_dir
    train_edges = numpy.load(os.path.join(FLAGS.output_dir, 'train.txt.npy'))
    for n1, n2 in list(train_edges):
      graph.add_edge(n1, n2)

  left_context = FLAGS.context * (not FLAGS.directed)
  print 'left_context = %i' % left_context
  SimulateWalks(
      graph, FLAGS.output_dir, num_walks=FLAGS.num_walks,
      context_right=FLAGS.context, context_left=left_context,
      walk_length=FLAGS.walk_length, p=N2V_P, q=N2V_Q)


if __name__ == '__main__':
  tf.app.run(main)
