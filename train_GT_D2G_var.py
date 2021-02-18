"""
Training scripts for GPT-GRNN Completed Version
Author: 
Create Date: Dec 16, 2020
"""

import time
import random
import statistics
from collections import Counter

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import sacred
from sacred.observers import FileStorageObserver

from utils import convert_adj_vec_to_matrix, get_sequence_lens_by_pointers, mask_generated_graph
from utils import length_penalty_func, calc_length_entropy
from model.data_loader import prepare_ingredients, collate_fn
from model.GPT_GRNN import GCNEncoder, GPTGRNNDecoderVariable, GraphClassifier
from model.GPT_GRNN import GPTGRNNDecoderSVar


# Sacred Setup
ex = sacred.Experiment('train_GT-D2G-var')
ex.observers.append(FileStorageObserver("logs/GT-D2G-var"))


@ex.config
def my_config():
    motivation = ""
    opt = {
           'gpu': False,
           'seed': 27,
           'corpus_type': '',    # 'yelp'|'dblp'|'nyt'
           'processed_pickle_path': '',
           'checkpoint_dir': '',
           'n_labels': {
               'nyt': 5,
               'yelp': 5,
               'yelp-3-class': 3,
               'dblp': 6
               },
           'epoch': 250,
           'epoch_warmup': 10,
           'early_stop_flag': False,
           'patience': 100,
           'batch_size': 128,
           'lr': 3e-4,
           'lr_scheduler_cosine_T_max': 64,
           'optimizer_weight_decay': 0.0,
           'lambda_cov_loss': 0.1,
           'beta_length_loss': 1.0,
           'clip_grad_norm': 5.0,
           'gptrnn_decoder_dropout': 0.3,
           'gcn_encoder_hidden_size': 128,
           'gcn_encoder_pooling': 'mean',
           'graph_rnn_num_layers': 2,
           'graph_rnn_hidden_size': 128,
           'edge_rnn_num_layers': 2,
           'edge_rnn_hidden_size': 16,
           'GPT_attention_unit': 10,
           'max_out_node_size': 15,
           'gumbel_tau_init': 5,
           'gumbel_tau_min': 0.5,
           'gumbel_tau_decay': 0.995,
           'gpt_grnn_variant': 'neigh-var',
           'gcn_classifier_hidden_size': 64,
           'pretrain_emb_name': 'glove.840B.300d.txt',
           'pretrain_emb_cache': None,
           'pretrain_emb_max_vectors': 160000,
           'yelp_senti_feat': False,
           'pretrain_emb_dropout': 0.0,
           'regular_loss_decay': 0.993,
           'length_penalty_flag': True,
           'anneal_cov_loss_flag': True,
           'anneal_length_loss_flag': True,
          }


@ex.automain
def train_model(opt, _run, _log):
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    th.manual_seed(opt['seed'])
    _log.info("The random seed has been set to %d globally" % (opt['seed']))

    # Sanity check
    if not opt['corpus_type'] or not opt['processed_pickle_path'] or not opt['checkpoint_dir']:
        _log.error('missing essential input arguments')
        exit(-1)
    n_labels = opt['n_labels'][opt['corpus_type']]
    lambda_cov_loss = opt['lambda_cov_loss']
    beta_length_loss = opt['beta_length_loss']

    # Load corpus
    batch_size = opt['batch_size']
    pickle_path = opt['processed_pickle_path']
    _log.info('[%s] Start loading %s corpus from %s' % (time.ctime(), opt['corpus_type'], pickle_path))
    train_set, val_set, test_set, vocab = prepare_ingredients(pickle_path, corpus_type=opt['corpus_type'],
                                                              pretrain_name=opt['pretrain_emb_name'],
                                                              emb_cache=opt['pretrain_emb_cache'],
                                                              max_vectors=opt['pretrain_emb_max_vectors'],
                                                              yelp_senti_feature=opt['yelp_senti_feat'])
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_iter = DataLoader(val_set, batch_size=min(1024, 8*batch_size), shuffle=False, collate_fn=collate_fn)
    _log.info('[%s] Load train, val, test sets Done, len=%d,%d,%d' % (time.ctime(),
              len(train_set), len(val_set), len(test_set)))

    # Build models
    pretrained_emb = vocab.vectors
    gcn_encoder = GCNEncoder(pretrained_emb, pretrained_emb.shape[1]+3, opt['gcn_encoder_hidden_size'],
                             opt['gcn_encoder_pooling'], opt['yelp_senti_feat'],
                             opt['pretrain_emb_dropout'])
    if opt['gpt_grnn_variant'] == 'path-var':
        gptrnn_decoder = GPTGRNNDecoderVariable(opt['gcn_encoder_hidden_size'], opt['GPT_attention_unit'],
                                                opt['max_out_node_size'], opt['graph_rnn_num_layers'],
                                                opt['graph_rnn_hidden_size'], opt['edge_rnn_num_layers'],
                                                opt['edge_rnn_hidden_size'],
                                                opt['gumbel_tau_init'], opt['gptrnn_decoder_dropout']
                                                )
        _log.info('use gt-d2g-path-var')
    elif opt['gpt_grnn_variant'] == 'neigh-var':
        gptrnn_decoder = GPTGRNNDecoderSVar(opt['gcn_encoder_hidden_size'], opt['GPT_attention_unit'],
                                            opt['max_out_node_size'], opt['graph_rnn_num_layers'],
                                            opt['graph_rnn_hidden_size'], opt['gumbel_tau_init'],
                                            opt['gptrnn_decoder_dropout']
                                            )
        _log.info('use gt-d2g-neigh-var')
    else:
        _log.info('gt-d2g-variant=%s is invalid' % (opt['gpt_grnn_variant']))
        exit(-1)
    gcn_classifier = GraphClassifier(opt['gcn_encoder_hidden_size'], opt['gcn_classifier_hidden_size'],
                                     n_labels)
    class_criterion = nn.CrossEntropyLoss()
    parameters = list(gcn_encoder.parameters()) + list(gptrnn_decoder.parameters()) \
        + list(gcn_classifier.parameters())
    optimizer = th.optim.Adam(parameters, opt['lr'], weight_decay=opt['optimizer_weight_decay'])
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt['lr_scheduler_cosine_T_max'])
    if opt['gpu']:
        gcn_encoder = gcn_encoder.cuda()
        gptrnn_decoder = gptrnn_decoder.cuda()
        gcn_classifier = gcn_classifier.cuda()
        class_criterion = class_criterion.cuda()

    # Start Epochs
    max_acc = 0.0
    max_nlen_entropy = 0.0
    for i_epoch in range(opt['epoch']):
        # Start Training
        gcn_encoder.train()
        gptrnn_decoder.train()
        gcn_classifier.train()
        train_loss = []
        train_class_loss = []
        train_cov_loss = []
        train_len_loss = []
        train_node_cnts = []
        all_pred = []
        all_gold = []
        for i_batch, batch in enumerate(train_iter):
            optimizer.zero_grad()
            batched_graph, nid_mappings, labels, docids = batch
            batch_size = labels.shape[0]
            if opt['gpu']:
                batched_graph = batched_graph.to('cuda:0')
                labels = labels.cuda()
            h, hg = gcn_encoder(batched_graph)
            pointer_argmaxs, cov_loss, encoder_out, adj_vecs, att_scores = gptrnn_decoder(batched_graph, h, hg)
            generated_node_lens = get_sequence_lens_by_pointers(pointer_argmaxs)   # (batch,)
            adj_matrix = convert_adj_vec_to_matrix(adj_vecs, add_self_loop=True)
            generated_nodes_emb = th.matmul(pointer_argmaxs.transpose(1, 2), encoder_out)  # batch*seq_l*hid
            generated_nodes_emb, adj_matrix = mask_generated_graph(generated_nodes_emb, adj_matrix, generated_node_lens)
            pred = gcn_classifier(generated_nodes_emb, adj_matrix)
            class_loss = class_criterion(pred, labels)
            if opt['length_penalty_flag']:
                length_loss = length_penalty_func(generated_node_lens, att_scores)
                loss = class_loss + lambda_cov_loss * cov_loss + beta_length_loss * length_loss
            else:
                loss = class_loss + lambda_cov_loss * cov_loss
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=opt['clip_grad_norm'], norm_type=2)
            optimizer.step()
            train_loss.append(loss.item())
            train_class_loss.append(class_loss.item())
            train_cov_loss.append(cov_loss.item())
            if opt['length_penalty_flag']:
                train_len_loss.append(length_loss.item())
            train_node_cnts.extend(generated_node_lens.tolist())
            all_gold.extend(labels.detach().tolist())
            all_pred.extend(th.argmax(pred, dim=1).detach().tolist())
        avg_loss = statistics.mean(train_loss)
        acc = (th.LongTensor(all_gold) == th.LongTensor(all_pred)).sum() / len(all_pred)
        nlen_entropy = calc_length_entropy(train_node_cnts)
        _run.log_scalar("train.loss", avg_loss, i_epoch)
        _run.log_scalar("train.class_loss", statistics.mean(train_class_loss), i_epoch)
        _run.log_scalar("train.cov_loss", statistics.mean(train_cov_loss), i_epoch)
        if opt['length_penalty_flag']:
            _run.log_scalar("train.length_loss", statistics.mean(train_len_loss), i_epoch)
        _run.log_scalar("train.node_cnt_avg", statistics.mean(train_node_cnts), i_epoch)
        _run.log_scalar("train.node_cnt_stdev", statistics.pstdev(train_node_cnts), i_epoch)
        _run.log_scalar("train.node_cnt_entropy", nlen_entropy)
        _run.log_scalar("train.acc", acc*100, i_epoch)
        _log.info('[%s] epoch#%d train Done, avg loss=%.5f, acc=%.2f' % (time.ctime(), i_epoch,
                                                                         avg_loss, acc * 100))
        _log.info('generated node cnt avg=%.3f, stdev=%.3f' % (statistics.mean(train_node_cnts),
                                                               statistics.pstdev(train_node_cnts)))

        # Start Validating
        gcn_encoder.eval()
        gptrnn_decoder.eval()
        gcn_classifier.eval()
        val_loss = []
        val_class_loss = []
        val_cov_loss = []
        val_len_loss = []
        val_node_cnts = []
        all_pred = []
        all_gold = []
        for i_batch, batch in enumerate(val_iter):
            batched_graph, nid_mappings, labels, docids = batch
            batch_size = labels.shape[0]
            if opt['gpu']:
                batched_graph = batched_graph.to('cuda:0')
                labels = labels.cuda()
            h, hg = gcn_encoder(batched_graph)
            pointer_argmaxs, cov_loss, encoder_out, adj_vecs, att_scores = gptrnn_decoder(batched_graph, h, hg)
            generated_node_lens = get_sequence_lens_by_pointers(pointer_argmaxs)   # (batch,)
            adj_matrix = convert_adj_vec_to_matrix(adj_vecs, add_self_loop=True)
            generated_nodes_emb = th.matmul(pointer_argmaxs.transpose(1, 2), encoder_out)  # batch*seq_l*hid
            generated_nodes_emb, adj_matrix = mask_generated_graph(generated_nodes_emb, adj_matrix, generated_node_lens)
            pred = gcn_classifier(generated_nodes_emb, adj_matrix)
            class_loss = class_criterion(pred, labels)
            if opt['length_penalty_flag']:
                length_loss = length_penalty_func(generated_node_lens, att_scores)
                loss = class_loss + lambda_cov_loss * cov_loss + beta_length_loss * length_loss
            else:
                loss = class_loss + lambda_cov_loss * cov_loss
            val_loss.append(loss.item())
            val_class_loss.append(class_loss.item())
            val_cov_loss.append(cov_loss.item())
            if opt['length_penalty_flag']:
                val_len_loss.append(length_loss.item())
            val_node_cnts.extend(generated_node_lens.tolist())
            all_gold.extend(labels.detach().tolist())
            all_pred.extend(th.argmax(pred, dim=1).detach().tolist())
        avg_loss = statistics.mean(val_loss)
        acc = (th.LongTensor(all_gold) == th.LongTensor(all_pred)).sum() / len(all_pred)
        nlen_entropy = calc_length_entropy(val_node_cnts)
        _run.log_scalar("eval.loss", avg_loss, i_epoch)
        _run.log_scalar("eval.class_loss", statistics.mean(val_class_loss), i_epoch)
        _run.log_scalar("eval.cov_loss", statistics.mean(val_cov_loss), i_epoch)
        if opt['length_penalty_flag']:
            _run.log_scalar("eval.length_loss", statistics.mean(val_len_loss), i_epoch)
        _run.log_scalar("eval.node_cnt_avg", statistics.mean(val_node_cnts), i_epoch)
        _run.log_scalar("eval.node_cnt_stdev", statistics.pstdev(val_node_cnts), i_epoch)
        _run.log_scalar("eval.node_cnt_entropy", nlen_entropy)
        _run.log_scalar("eval.acc", acc*100, i_epoch)
        _log.info('[%s] epoch#%d validation Done, avg loss=%.5f, acc=%.2f' % (time.ctime(), i_epoch,
                                                                              avg_loss, acc * 100))
        _log.info('generated node cnt avg=%.3f, stdev=%.3f' % (statistics.mean(val_node_cnts),
                                                               statistics.pstdev(val_node_cnts)))
        if i_epoch > opt['epoch_warmup']:
            if acc > max_acc:
                max_acc = acc
                save_path = '%s/exp%s_%s.best.ckpt' % (opt['checkpoint_dir'], _run._id, opt['corpus_type'])
                _log.info('Achieve best acc, store model into %s' % (save_path))
                th.save({'gcn_encoder': gcn_encoder.state_dict(),
                         'gptrnn_decoder': gptrnn_decoder.state_dict(),
                         'gcn_classifier': gcn_classifier.state_dict()
                         }, save_path)
                patience = 0
            else:
                patience += 1
            # early stop
            if opt['early_stop_flag'] and patience > opt['patience']:
                _log.info('Achieve best acc=%.2f, early stop at epoch #%d' % (max_acc*100, i_epoch))
                exit(0)
            # check model with highest length diversity, i.e. entropy
            if nlen_entropy >= max_nlen_entropy:
                max_nlen_entropy = nlen_entropy
                l_dist = Counter()
                for cnt in val_node_cnts:
                    l_dist[cnt] += 1
                l_dist = sorted(l_dist.items(), key=lambda _: _[0])
                _log.info("Achieve max nlen_entropy=%.2f: %s" % (max_nlen_entropy, l_dist))
        # scheduler
        scheduler.step()
        # Anneal gumbel-softmax tau
        gptrnn_decoder.tau = max(gptrnn_decoder.tau * opt['gumbel_tau_decay'], opt['gumbel_tau_min'])
        # Anneal regularization loss
        if opt['anneal_cov_loss_flag']:
            lambda_cov_loss *= opt['regular_loss_decay']
        if opt['anneal_length_loss_flag']:
            beta_length_loss *= opt['regular_loss_decay']
