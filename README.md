This is the codebase for reproducing results of KDD'2021 submission: "Weakly Supervised Concept Map Generationthrough Task-Guided Graph Translation".

## Prerequisites

```
python==3.7.9
```

For library requirements, please refer to `./requirements.txt`.

## Data

**Pre-processed Graphs**

The NLP pipeline derived initial concept maps can be download from https://figshare.com/s/e85a87db24d01a245e93. Put it under the project root directory and decompress it directly.
Then three `*.pickle.gz` files would reside under `./data/`. (No need to decompress *.pickle.gz files)

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
