#!/bin/bash

# Reproduce NYT Result
# `with` syntax is introduced by sacred(https://github.com/IDSIA/sacred)
# python test_GT_D2G.py with 'config_path="checkpoints/GT-D2G-var/nyt_seed27.best.config"' \
#      'checkpoint_path="checkpoints/GT-D2G-var/nyt_seed27.best.ckpt"'
# Reproduce AMiner Result
python test_GT_D2G.py with 'config_path="checkpoints/GT-D2G-var/dblp_seed27.best.config"' \
     'checkpoint_path="checkpoints/GT-D2G-var/dblp_seed27.best.ckpt"'
# Reproduce Yelp Result
python test_GT_D2G.py with 'config_path="checkpoints/GT-D2G-var/yelp_seed27.best.config"' \
     'checkpoint_path="checkpoints/GT-D2G-var/yelp_seed27.best.ckpt"'
