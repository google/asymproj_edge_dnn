# Learning Edge Representations via Low-Rank Asymmetric Projections

Implementation of [ACM CIKM 2017 paper](https://arxiv.org/abs/1705.05615)
_Learning Edge Representation via Low-Rank Asymmetric Projections_. As described
below, this repository includes:

1. Code to process a graph (i.e. create training files).
1. Code to train node embeddings and edge function, using our method, and
   evaluation code on link prediction tasks.
1. Dataset files, that are used in our paper.

If you use this code, then you should:

1. Note that this is **not** an official Google product. Please direct your
   questions to the main author (Sami Abu-El-Haija).
1. Consider citing our work, using the following bibtex:
   
       @INPROCEEDINGS{asymproj,
         authors = {Sami Abu-El-Haija AND Bryan Perozzi AND Rami Al-Rfou},
         title = {Learning Edge Representations via Low-Rank Asymmetric Projections},
         booktitle = {ACM International Conference on Information and Knowledge Management (CIKM)},
         year = {2017},
       }

## Overview of files

* [`create_dataset_arrays.py`](./create_dataset_arrays.py): Processes a graph,
  as an edge-list file, and produces training files.
* [`deep_edge_trainer.py`](./deep_edge_trainer.py): Processes the output of
  `create_dataset_arrays.py` and trains a model, continuously evaluating the
  model on the test partition.
* [`datasets/`](./datasets): Directory containing datasets used in our paper.
  The original datasets come from [Stanford SNAP](http://snap.stanford.edu/data)
  and [BioGrid](https://thebiogrid.org). Nonetheless, we release our train/test
  splits for others to replicate our results.

## How to use

To use, you must first create dataset files
(using [`create_dataset_arrays.py`](./create_dataset_arrays.py)), then train the node
embeddings and the edge function
(using [`deep_edge_trainer.py`](./deep_edge_trainer.py)). The following two
subsections explain how to use these two python scripts.

### Create Trainer Files

The input to our method is an edge-list (readable by
[`nx.read_edgelist`](https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.readwrite.edgelist.read_edgelist.html)).
In particular, assume that your file `/path/to/graph.txt` contains lines like:

    n1 n2

where `n1` and `n2` are node IDs, which can be strings or integers. The line
indicates edge `n1-n2` (if graph is undirected) or `n1->n2` (if graph is
directed). Assume that the above `graph.txt` contains `|E|` lines. You can
generate dataset files by running:

    python create_dataset_arrays.py --input /path/to/graph.txt --output ~/asymproj/datasets/my_graph

Which creates directory `~/asymproj/datasets/my_graph`, writing files:

1. `train.txt.npy` and `test.txt.npy`: int32 numpy arrays, each of size
   `(|E|/2, 2)`, containing train and test edges. They union to graph.txt,
   except that the node IDs are _renumbered_ to be `[0, 1, 2, ..., |V| - 1]`.
1. `train.neg.txt.npy` and `test.neg.txt.npy`: int32 numpy arrays, each of size
   `(|E|/2, 2)`, containing negative edges for training and testing. The first
   is sampled from the compliment of `train.txt.npy` and the second is sampled
   from the compliment of `union(train.txt.npy, test.txt.npy)`.
1. `test.directed.neg.txt.npy` only if `--directed` flag is set. It is set to
   contents of `test.neg.txt.npy` plus all `(u, v)` if `(v, u)` is an edge but
   `(u, v)` is not. These are _harder negatives_ that evaluates the capability
   of the edge representation to learn edge direction.
1. `index.pkl`: Pickle serialization of python dictionary containing graph
   statistics, as well as key 'index' that is a mapping of
   `original node ID -> renumbered node ID`.
1. `train.neg_per_node.txt.npy`: pre-sampled 20 negative nodes per node, used
   for noise estimation.
1. `train.pairs.<i>.txt.npy`: Training pairs. Sampled using random walks.
   Presence of pair `(u, v)`, indicates that `u` and `v` appeared in a random
   walk, within `--context` hops.

#### Pre-processed dataset files

If you are using pre-processed dataset files, with `{train, test}` and 
`{positive, negative}` edges generated as numpy files, you can simulate walks
on train positives and create train negatives using by appending flag
`--only_simulate_walks` to binary `create_dataset_arrays.py`. For example, if
you want to simulate walks for the PPI dataset (see [datasets](./datasets/)),
then you can run the command:

    python create_dataset_arrays.py --output_dir /path/to/asymproj/datasets/ppi --only_simulate_walks


### Run Trainer Code

The training binary `deep_edge_trainer.py` trains our model. There are many
flags that customize the architecture, including size of node embedding
(`--embed_dim`), latent dimensions of the deep neural network (`--dnn_dims`),
and the low-rank projection dimension (`--projection_dim`). The dataset
directory `--dataset_dir` must contain all files produced by
`create_dataset_arrays.py`. During training, the trainer outputs the trained
model onto subdirectory `dumps`. The name of the model files are automatically
captured from the hyper-parameters (e.g. dimensions of embedding, dnn,
projection). Setting flag `--restore=0` will force to the trainer to start from
scratch. Otherwise, if flag is ommitted, the trainer will continue training the
model from the last saved state.

If you simulated walks on PPI, you can train a model with:

    python deep_edge_trainer.py --dataset_dir /path/to/asymproj/datasets/ppi

The above command will print many lines like this:

    @0 (790) test/train Best=0.80/0.85 cur=0.81/0.84. - test.d100_f100x100_g32

where `@0` is epoch `0`, `790` is the batch ID, and the metrics
`Best=0.80/0.85` and `cur=0.81/0.84`, respectively, are the best
and current accuracies like `test/train` accuracies. The `Best` is taken for
_best train_. We perform model selection at the training set. In other words, we
select model parameters at peak train accuracy. Please view the flags of
[`deep_edge_trainer.py`](./deep_edge_trainer.py) to customize model
hyper-parameters.

