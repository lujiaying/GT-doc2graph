#!/bin/bash

# Train GT-D2G-var
# AMiner
# python train_GT_D2G_var.py with 'motivation="neigh-var"' \
#     'opt.gpu=True' 'opt.seed=27' \
#     'opt.gpt_grnn_variant="neigh-var"' 'opt.anneal_cov_loss_flag=False' \
#     'opt.lambda_cov_loss=0.2' 'opt.pretrain_emb_dropout=0.5' \
#     'opt.max_out_node_size=10' 'opt.beta_length_loss=5.0' \
#     'opt.epoch=500' 'opt.epoch_warmup=10' \
#     'opt.corpus_type="dblp"' 'opt.processed_pickle_path="data/dblp.win5.pickle.gz"' \
#     'opt.batch_size=64' 'opt.lr=3e-4' 'opt.optimizer_weight_decay=1e-4' \
#     'opt.checkpoint_dir="checkpoints/GT-D2G-var"'

# Train GT-D2G-neigh
# Yelp
python train_GT_D2G_neigh.py with 'motivation="neigh"' \
    'opt.gpu=True' 'opt.seed=27' \
    'opt.lambda_cov_loss=1.0' 'opt.epoch=500'\
    'opt.corpus_type="yelp-3-class"' 'opt.processed_pickle_path="data/yelp.sentiment_centric.win5.pickle.gz"' \
    'opt.batch_size=128' 'opt.lr=3e-4' 'opt.optimizer_weight_decay=1e-4'\
    'opt.pretrain_emb_dropout=0.5' \
    'opt.pretrain_emb_name="glove.840B.restaurant.400d.vec"' 'opt.pretrain_emb_cache=".vector_cache"'\
    'opt.pretrain_emb_max_vectors=320000' \
    'opt.checkpoint_dir="checkpoints/GT-D2G-neigh"'

# Train GT-D2G-path
# NYT
