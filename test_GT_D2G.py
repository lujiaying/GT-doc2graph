"""
Test scripts
Author:
Create Date: Dec 10, 2020
"""

import os
import time
import json
import random
import statistics
import pickle

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import sacred

from utils import convert_adj_vec_to_matrix, get_sequence_lens_by_pointers, mask_generated_graph
from model.data_loader import prepare_ingredients, collate_fn
from model.GPT_GRNN import GCNEncoder, GraphClassifier
from model.GPT_GRNN import GPTGRNNDecoderS, GPTGRNNDecoder, GPTGRNNDecoderVariable
from model.GPT_GRNN import GPTGRNNDecoderSVar


# Sacred Setup
ex = sacred.Experiment('test_GT-D2G')


@ex.config
def my_config():
    config_path = ''
    checkpoint_path = ''
    gumbel_tau = 1e5   # high temperaute for evaluate


@ex.automain
def test_model(config_path, checkpoint_path, gumbel_tau, _run, _log):
    if not config_path or not checkpoint_path:
        _log.error('missing arg=config_path | checkpoint_path')
        exit(-1)

    # Load config
    _log.info('Load config from %s' % (config_path))
    with open(config_path) as fopen:
        loaded_cfg = json.load(fopen)
        opt = loaded_cfg['opt']
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    th.manual_seed(opt['seed'])
    n_labels = opt['n_labels'][opt['corpus_type']]
    # recover newly added params
    pretrain_emb_name = opt.get('pretrain_emb_name', 'glove.840B.300d.txt')
    pretrain_emb_cache = opt.get('pretrain_emb_cache', None)
    pretrain_emb_max_vectors = opt.get('pretrain_emb_max_vectors', 160000)
    gcn_encoder_pooling = opt.get('gcn_encoder_pooling', 'mean')
    yelp_senti_feat = opt.get('yelp_senti_feat', False)
    pretrain_emb_dropout = opt.get('pretrain_emb_dropout', 0.0)

    # Load corpus
    batch_size = opt['batch_size']
    pickle_path = opt['processed_pickle_path']
    _log.info('[%s] Start loading %s corpus from %s' % (time.ctime(), opt['corpus_type'], pickle_path))
    train_set, val_set, test_set, vocab = prepare_ingredients(pickle_path, corpus_type=opt['corpus_type'],
                                                              pretrain_name=pretrain_emb_name,
                                                              emb_cache=pretrain_emb_cache,
                                                              max_vectors=pretrain_emb_max_vectors,
                                                              yelp_senti_feature=yelp_senti_feat)
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    _log.info('[%s] Load train, val, test sets Done, len=%d,%d,%d' % (time.ctime(),
              len(train_set), len(val_set), len(test_set)))

    # Build models
    pretrained_emb = vocab.vectors
    gcn_encoder = GCNEncoder(pretrained_emb, pretrained_emb.shape[1]+3, opt['gcn_encoder_hidden_size'],
                             gcn_encoder_pooling, yelp_senti_feat, pretrain_emb_dropout)
    if opt['gpt_grnn_variant'] == 'simple':
        gptrnn_decoder = GPTGRNNDecoderS(opt['gcn_encoder_hidden_size'], opt['GPT_attention_unit'],
                                         opt['max_out_node_size'], gumbel_tau)
    elif opt['gpt_grnn_variant'] == 'complete':
        gptrnn_decoder = GPTGRNNDecoder(opt['gcn_encoder_hidden_size'], opt['GPT_attention_unit'],
                                        opt['max_out_node_size'], opt['graph_rnn_num_layers'],
                                        opt['graph_rnn_hidden_size'], opt['edge_rnn_num_layers'],
                                        opt['edge_rnn_hidden_size'],
                                        gumbel_tau, opt['gptrnn_decoder_dropout']
                                        )
    elif opt['gpt_grnn_variant'] == 'variable':
        gptrnn_decoder = GPTGRNNDecoderVariable(opt['gcn_encoder_hidden_size'], opt['GPT_attention_unit'],
                                                opt['max_out_node_size'], opt['graph_rnn_num_layers'],
                                                opt['graph_rnn_hidden_size'], opt['edge_rnn_num_layers'],
                                                opt['edge_rnn_hidden_size'],
                                                opt['gumbel_tau_init'], opt['gptrnn_decoder_dropout']
                                                )
    elif opt['gpt_grnn_variant'] == 'neigh-var':
        gptrnn_decoder = GPTGRNNDecoderSVar(opt['gcn_encoder_hidden_size'], opt['GPT_attention_unit'],
                                            opt['max_out_node_size'], opt['graph_rnn_num_layers'],
                                            opt['graph_rnn_hidden_size'], opt['gumbel_tau_init'],
                                            opt['gptrnn_decoder_dropout']
                                            )
    else:
        _log.error('invalid gpt_grnn_variant=%s, expected "simple"|"complete"')
        exit(-1)
    gcn_classifier = GraphClassifier(opt['gcn_encoder_hidden_size'], opt['gcn_classifier_hidden_size'],
                                     n_labels)
    class_criterion = nn.CrossEntropyLoss()
    # Load checkpoints
    checkpoint = th.load(checkpoint_path)
    gcn_encoder.load_state_dict(checkpoint['gcn_encoder'])
    gptrnn_decoder.load_state_dict(checkpoint['gptrnn_decoder'])
    gcn_classifier.load_state_dict(checkpoint['gcn_classifier'])
    _log.info('Load state_dict from %s' % (checkpoint_path))
    if opt['gpu']:
        gcn_encoder = gcn_encoder.cuda()
        gptrnn_decoder = gptrnn_decoder.cuda()
        gcn_classifier = gcn_classifier.cuda()
        class_criterion = class_criterion.cuda()
    gcn_encoder.eval()
    gptrnn_decoder.eval()
    gcn_classifier.eval()
    val_node_cnts = []
    all_pred = []
    all_gold = []
    with th.no_grad():
        for i_batch, batch in enumerate(test_iter):
            batched_graph, nid_mappings, labels, docids = batch
            batch_size = labels.shape[0]
            if opt['gpu']:
                batched_graph = batched_graph.to('cuda:0')
                labels = labels.cuda()
            h, hg = gcn_encoder(batched_graph)
            if opt['gpt_grnn_variant'] in ['variable', 'neigh-var']:
                pointer_argmaxs, cov_loss, encoder_out, adj_vecs, att_scores = gptrnn_decoder(batched_graph, h, hg)
                generated_node_lens = get_sequence_lens_by_pointers(pointer_argmaxs)   # (batch,)
                adj_matrix = convert_adj_vec_to_matrix(adj_vecs, add_self_loop=True)
                generated_nodes_emb = th.matmul(pointer_argmaxs.transpose(1, 2), encoder_out)  # batch*seq_l*hid
                generated_nodes_emb, adj_matrix = mask_generated_graph(generated_nodes_emb, adj_matrix,
                                                                       generated_node_lens)
                pred = gcn_classifier(generated_nodes_emb, adj_matrix)
                val_node_cnts.extend(generated_node_lens.tolist())
            else:
                pointer_argmaxs, cov_loss, encoder_out, adj_vecs = gptrnn_decoder(batched_graph, h, hg)
                adj_matrix = convert_adj_vec_to_matrix(adj_vecs, add_self_loop=True)
                generated_nodes_emb = th.matmul(pointer_argmaxs.transpose(1, 2), encoder_out)  # batch*seq_l*hid
                pred = gcn_classifier(generated_nodes_emb, adj_matrix)
            all_gold.extend(labels.detach().tolist())
            all_pred.extend(th.argmax(pred, dim=1).detach().tolist())
    acc = (th.LongTensor(all_gold) == th.LongTensor(all_pred)).sum() / len(all_pred)
    _log.info('[%s] acc=%.2f, ' % (time.ctime(), acc*100))
    if opt['gpt_grnn_variant'] in ['variable', 'neigh-var']:
        _log.info('generated node cnt avg=%.3f, stdev=%.3f' % (statistics.mean(val_node_cnts),
                                                               statistics.pstdev(val_node_cnts)))
