# Weakly Supervised Concept Map Generation through Task-Guided Graph Translation

## GT-D2G

This is the codebase for reproducing results of KDD'2021 submission: "Weakly Supervised Concept Map Generation through Task-Guided Graph Translation".

![Proposed Framework](imgs/)

![Graph Translator](imgs/)

## Prerequisites

```
python==3.7.9
```

For library requirements, please refer to `./requirements.txt`. (You may replace [PyTorch](https://pytorch.org/) and [dgl](https://www.dgl.ai/pages/start.html) to CPU version)

## Data

**Pre-processed Graphs**

The NLP pipeline derived initial concept maps can be download from https://figshare.com/s/e85a87db24d01a245e93.    
Put it under the project root directory and decompress it directly. Then three `*.pickle.gz` files would reside under `./data/`. (No need to decompress *.pickle.gz files)

```
./data
|-- dblp.txt
|-- dblp.win5.pickle.gz
|-- nyt.txt
|-- nyt.win5.pickle.gz
|-- yelp.txt
|-- yelp.sentiment_centric.win5.pickle.gz
```

**Checkpoints**

- `GT-D2G-path`:
- `GT-D2G-neigh`:
- `GT-D2G-var`: https://figshare.com/s/1bea7883754b4fce3b7f

Please download gziped checkpoint files using the above urls, and decompress them under `./checkpoints` folder.

**Pre-trained Word Embeddings**

*GT-D2G* relies GloVe embedding. Download `glove.840B.300d` from https://nlp.stanford.edu/projects/glove/, and put it under `./.vector_cache`.  
For yelp dataset, we get the best performan using a hybrid of GloVe and [restaurant embedding](https://howardhsu.github.io/dataset/), which can be download from https://figshare.com/s/3eb1271fef804e9ab8fe.

```
./.vector_cache
|--glove.840B.300d.txt
|--glove.840B.restaurant.400d.vec
```
