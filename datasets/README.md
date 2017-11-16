# Datasets

Our work utilizes a number of graph datasets provided from
[Stanford SNAP](http://snap.stanford/edu/data), as well as a Protein-Protein
Interaction graph dataset from [BioGrid](https://thebiogrid.org).

We packaged our train/test splits onto:
[http://sami.haija.org/graph/datasets.tgz](http://sami.haija.org/graph/datasets.tgz),
each dataset is in a directory. The directories are:

1. `ca-AstroPh`
2. `ca-HepTh`
3. `soc-epinions`
4. `soc-facebook`
5. `wiki-vote`
6. `ppi`

Each directory contains the files:

1. `index.pkl`: Python pickle of a dictionary, containing key `index` that is a
   mapping from original node ID into integer in range `[0, |V| - 1]`, where
   `|V|` are the number of nodes in the graph. This **renumbering** is important
   to make sure that node IDs can be represented in an embedding matrix, where
   the node ID refers to a row in the matrix. All files below uses these
   integers.
1. `train.txt.npy`: Positive training edges. Integer numpy array of size
   `(|E|/2, 2)`, where `|E|` is the number of edges in the original graph, and
   therefore `|E|/2` is the number of training edges.
1. `train.neg.txt.npy`: Negative training edges.
   Integer numpy array of size `(|E|/2, 2)`.
1. `test.txt.npy`: Positive test edges.
   Integer numpy array of size `(|E|/2, 2)`.
1. `test.neg.txt.npy`: Negative test edges.
   Integer numpy array of size `(|E|/2, 2)`.

