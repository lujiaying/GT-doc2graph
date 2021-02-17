#!/bin/bash

# ** GT-D2G-var **
# Reproduce NYT Result
# `with` syntax is introduced by sacred(https://github.com/IDSIA/sacred)
# python test_GT_D2G.py with 'config_path="checkpoints/GT-D2G-var/nyt_seed27.best.config"' \
#      'checkpoint_path="checkpoints/GT-D2G-var/nyt_seed27.best.ckpt"'
# Reproduce AMiner Result
# python test_GT_D2G.py with 'config_path="checkpoints/GT-D2G-var/dblp_seed27.best.config"' \
#      'checkpoint_path="checkpoints/GT-D2G-var/dblp_seed27.best.ckpt"'
# # Reproduce Yelp Result
# python test_GT_D2G.py with 'config_path="checkpoints/GT-D2G-var/yelp_seed27.best.config"' \
#      'checkpoint_path="checkpoints/GT-D2G-var/yelp_seed27.best.ckpt"'

# ** GT-D2G-neigh **
# Reproduce NYT Result
python test_GT_D2G.py with 'config_path="checkpoints/GT-D2G-neigh/nyt_seed27.best.config"' \
     'checkpoint_path="checkpoints/GT-D2G-neigh/nyt_seed27.best.ckpt"'
# Reproduce AMiner Result
python test_GT_D2G.py with 'config_path="checkpoints/GT-D2G-neigh/dblp_seed27.best.config"' \
     'checkpoint_path="checkpoints/GT-D2G-neigh/dblp_seed27.best.ckpt"'
# Reproduce Yelp Result
python test_GT_D2G.py with 'config_path="checkpoints/GT-D2G-neigh/yelp_seed27.best.config"' \
     'checkpoint_path="checkpoints/GT-D2G-neigh/yelp_seed27.best.ckpt"'
