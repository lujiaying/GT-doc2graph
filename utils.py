"""
Utility functions
Author:
Create Date: Dec 8, 2020
"""

import math
from collections import Counter

import torch as th
import networkx as nx
from allennlp.nn.util import get_mask_from_sequence_lengths


def convert_adj_vec_to_matrix(adj_vecs: th.Tensor, add_self_loop: bool) -> th.Tensor:
    # adj_vecs: batch * max_node-1 * max_node
    batch_size = adj_vecs.shape[0]
    n = adj_vecs.shape[2]
    adj_matrix = adj_vecs.new_zeros((batch_size, n, n))

    for j in range(1, n):
        for i in range(j):
            adj_matrix[:, i, j] += adj_vecs[:, i, j]
            adj_matrix[:, j, i] += adj_vecs[:, i, j]
    if add_self_loop:
        # self_loop_matrix = th.diag(th.ones(n)).unsqueeze(0).repeat(batch_size, 1, 1)
        self_loop_matrix = th.diag(adj_vecs.new_ones(n)).unsqueeze(0).repeat(batch_size, 1, 1)
        adj_matrix += self_loop_matrix
    return adj_matrix


def generate_pred_graphs(docids: list, nid_mappings: list,
                         pointer_argmaxs: th.Tensor, adj_matrix: th.Tensor) -> list:
    # pointer_argmaxs: batch * max_nnodes * max_seq_len
    # adj_matrix: batch * max_seq_len * max_seq_len, with self_loop
    max_seq_len = adj_matrix.shape[-1]
    G_list = []
    chosen_node_idxs = th.argmax(pointer_argmaxs, dim=1)  # batch*max_seq_len
    for i_batch in range(len(docids)):
        G = nx.Graph()
        chosen_nodes = chosen_node_idxs[i_batch, :]
        nid_mention_map = {v: k for k, v in nid_mappings[i_batch].items()}
        G_local_id_mention_map = {}
        for nid in chosen_nodes:
            mention = nid_mention_map[nid.item()]
            G_local_id_mention_map[len(G_local_id_mention_map)] = mention
            G.add_node(mention)
        for i in range(max_seq_len-1):
            for j in range(i+1, max_seq_len):
                weight = adj_matrix[i_batch, i, j].item()
                mention_i = G_local_id_mention_map[i]
                mention_j = G_local_id_mention_map[j]
                G.add_edge(mention_i, mention_j, weight=weight)
        G.graph['docid'] = docids[i_batch]
        G_list.append(G)
    return G_list


def get_sequence_lens_by_pointers(pointers: th.Tensor) -> th.Tensor:
    """
    Args:
        pointers: batch*max_nnode+1*max_seq_len
    """
    batch_size = pointers.shape[0]
    max_seq_len = pointers.shape[2]
    seq_lens = pointers.new_zeros(batch_size, dtype=th.long)
    for i_batch in range(batch_size):
        chosen_nidx = pointers[i_batch, :, :].argmax(dim=0).tolist()  # max_seq_len
        if 0 in chosen_nidx:
            cur_len = chosen_nidx.index(0)
        else:
            cur_len = max_seq_len
        seq_lens[i_batch] = cur_len
    return seq_lens


def mask_generated_graph(generated_nodes_emb: th.Tensor, adj_matrix: th.Tensor, seq_lens: th.Tensor) -> tuple:
    """
    Args:
        generated_nodes_emb: batch*max_seq_len*hid
        adj_matrix: batch*max_seq_len*max_seq_len
        seq_lens: batch
    """
    batch_size = generated_nodes_emb.shape[0]
    max_l = generated_nodes_emb.shape[1]
    mask = get_mask_from_sequence_lengths(seq_lens, max_length=max_l)   # batch*max_len
    generated_nodes_emb = generated_nodes_emb.permute(2, 0, 1).masked_fill(~mask, 0.0)
    generated_nodes_emb = generated_nodes_emb.permute(1, 2, 0)
    matrix_mask = []
    for i in range(batch_size):
        cl = seq_lens[i].item()
        mask = th.block_diag(adj_matrix.new_ones(cl, cl), adj_matrix.new_zeros(max_l-cl, max_l-cl))  # max_l*max_l
        matrix_mask.append(mask)
    matrix_mask = th.stack(matrix_mask, dim=0) >= 1.0    # batch*max_seq_l*max_seq_l
    adj_matrix.masked_fill_(~matrix_mask, 0.0)
    return generated_nodes_emb, adj_matrix


def cov_loss_func(cov_vecs: th.Tensor, att_scores: th.Tensor,
                  seq_lens: th.Tensor) -> th.Tensor:
    """
    Args:
        cov_vecs:    batch*max_nnode+1*max_seq_len
        att_scores:  batch*max_nnode+1*max_seq_len
        seq_lens:    batch
    """
    # batch_size = cov_vecs.shape[0]
    # max_nnode = cov_vecs.shape[1]
    max_l = cov_vecs.shape[2]
    cov_loss = cov_vecs.new_tensor(0.0)
    mask = get_mask_from_sequence_lengths(seq_lens, max_length=max_l)   # batch*max_len
    cov_vecs = cov_vecs.permute(1, 0, 2).masked_fill(~mask, 0.0)
    cov_vecs = cov_vecs.permute(1, 0, 2)    # batch*max_nnode*max_len
    for i in range(max_l):
        # cur_cov_vec = cov_vecs[:, :, i]  # batch*max_nnode
        cov_loss += th.minimum(att_scores[:, :, i], cov_vecs[:, :, i]).sum()
    # TODO: consider whether change seq_lens.sum() to a fixed divider
    # or even collect all generated pointers
    cov_loss = cov_loss / seq_lens.sum()
    return cov_loss


def length_penalty_func(seq_lens: th.Tensor, att_scores: th.Tensor) -> th.Tensor:
    """
    Args:
        seq_lens: batch
        att_scores: batch*max_nnode+1*max_seq_len
    """
    batch_size = att_scores.shape[0]
    max_l = att_scores.shape[2]
    # Length Penalty: apply decay loss on length, the shorter the higher.
    # RBF Kernel as Penalty Function
    # inspired by ICLR20', https://openreview.net/pdf?id=SylkzaEYPS
    sigma = 4
    gamma = 1 / 2 / (sigma ** 2)
    t_prime = 0
    penalty = [math.exp(-gamma * ((t-t_prime)**2)) for t in range(max_l)]
    # [1.0000, 0.9692, 0.8825, 0.7548, 0.6065, 0.4578, 0.3247, 0.2163, 0.1353, 0.0796]
    penalty = att_scores.new_tensor(penalty)  # max_seq_len
    p = att_scores[:, 0, :].squeeze(1)   # batch*max_seq_len
    length_loss = p * penalty
    length_loss = length_loss.sum() / (batch_size * max_l)
    return length_loss


def calc_length_entropy(node_cnts: list) -> float:
    entropy = 0.0
    l_dist = Counter()
    for cnt in node_cnts:
        l_dist[cnt] += 1
    for k, v in l_dist.items():
        p = v / len(node_cnts)
        entropy += (-p * math.log2(p))
    return entropy
