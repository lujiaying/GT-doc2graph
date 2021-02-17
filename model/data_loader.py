"""
Data Loader for Document Classification Datasets
Author:
Create Date: Nov 26, 2020
"""

import random
import pickle

import torchtext
from torch.utils.data import Dataset
import torch as th
import dgl


class DocClassificationDataset(Dataset):
    def __init__(self, graphs: list, vocab: torchtext.vocab.Vocab, node_max_token: int = 8,
                 corpus_type: str = 'nyt', yelp_senti_feature: bool = False,
                 self_loop_flag: bool = True) -> None:
        self.graphs = []          # store dgl.DGLGraph
        self.nid_mappings = []    # store mention to nid mapping
        self.labels = []          # store classification labels
        self.docids = []          # starts from 1
        epsilon = 1e-6     # avoid div zero
        for nxg in graphs:
            self.docids.append(nxg.graph['docid'])
            if corpus_type == 'yelp':
                self.labels.append(nxg.graph['class'] - 1)   # yelp 1-5
            elif corpus_type == 'dblp':
                category = nxg.graph['class'] if nxg.graph['class'] != 6 else 5  # 0,1,2,3,4,6
                self.labels.append(category)
            else:
                self.labels.append(nxg.graph['class'])
            nid_mapping = {}   # mention -> nid
            freqs = []    # node frequency
            loc_f = []    # node first position
            loc_l = []    # node last position
            phrase_tids = th.ones(nxg.number_of_nodes(), node_max_token,
                                  dtype=th.long)   # node tokens, id=1:'<pad>'
            phrase_lens = []
            if yelp_senti_feature:
                pos_levels = []   # node postive sentiment level
                neg_levels = []   # node negative sentiment level
            for idx, n in enumerate(nxg.nodes()):
                nid_mapping[n] = idx
                freqs.append(nxg.nodes[n]['freq'])
                loc_f.append(min([_[0] for _ in nxg.nodes[n]['offsets']]))
                loc_l.append(max([_[1] for _ in nxg.nodes[n]['offsets']]))
                for i, t in enumerate(n.split(' ')):
                    if i >= node_max_token:
                        break
                        # print('For #%d graph, phrase:"%s" exceed max token len' % (idx, n))
                    phrase_tids[idx][i] = vocab.stoi[t]
                phrase_lens.append(min(i+1, node_max_token))
                if yelp_senti_feature:
                    pos_levels.append(nxg.nodes[n]['senti_pos_l'])
                    neg_levels.append(nxg.nodes[n]['senti_neg_l'])
            src_nids = []
            dst_nids = []
            for edges in nxg.edges():
                nid_0 = nid_mapping[edges[0]]
                nid_1 = nid_mapping[edges[1]]
                src_nids.append(nid_0)
                dst_nids.append(nid_1)
                src_nids.append(nid_1)  # undirected graph
                dst_nids.append(nid_0)
            g = dgl.graph((src_nids, dst_nids), num_nodes=nxg.number_of_nodes())
            if self_loop_flag:
                g = dgl.add_self_loop(g)
            g.ndata['f'] = th.tensor(freqs, dtype=th.short)   # freq
            g.ndata['lf'] = th.tensor(loc_f, dtype=th.int)    # location first
            g.ndata['ll'] = th.tensor(loc_l, dtype=th.int)    # location last
            g.ndata['ml'] = th.tensor(phrase_lens, dtype=th.int)   # mention lengths
            g.ndata['p'] = phrase_tids                        # phrase token ids
            if yelp_senti_feature:
                g.ndata['pos'] = th.tensor(pos_levels, dtype=th.long)  # sentiment pos
                g.ndata['neg'] = th.tensor(neg_levels, dtype=th.long)  # sentiment neg
            # 0-1 normalize
            g.ndata['f'] = (g.ndata['f'] - g.ndata['f'].min()) / ((g.ndata['f']).max() - g.ndata['f'].min() + epsilon)
            g.ndata['lf'] = (g.ndata['lf'] - g.ndata['lf'].min()) / (g.ndata['lf'].max() - g.ndata['lf'].min() + epsilon)
            g.ndata['ll'] = (g.ndata['ll'] - g.ndata['ll'].min()) / (g.ndata['ll'].max() - g.ndata['ll'].min() + epsilon)
            self.nid_mappings.append(nid_mapping)
            self.graphs.append(g)
            """
            # TODO: debug
            if len(self.graphs) > 3000:
                break
            # TODO: End debug
            """

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple:
        return self.graphs[idx], self.nid_mappings[idx], self.labels[idx], self.docids[idx]


def collate_fn(data: list) -> tuple:
    # return map(list, zip(*data))
    graphs, batched_nid_mappings, batched_labels, docids = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    return batched_graph, batched_nid_mappings, th.LongTensor(batched_labels), docids


def get_vocab(train_graphs: list, val_graphs: list, pretrained_vec: torchtext.vocab.Vectors,
              sos_eos_flag: bool = False) -> torchtext.vocab.Vocab:
    counter = {k: 1 for k in pretrained_vec.stoi}
    counter['<unk>'] = 3
    counter['<pad>'] = 1
    if sos_eos_flag is True:
        counter['<sos>'] = 1
        counter['<eos>'] = 1
    for G in train_graphs:
        for n in G.nodes():
            for word in n.split(' '):
                if word in counter:
                    counter[word] += 1
                else:
                    counter[word] = 1
    for G in val_graphs:
        for n in G.nodes():
            for word in n.split(' '):
                if word in counter:
                    counter[word] += 1
                else:
                    counter[word] = 1
    v = torchtext.vocab.Vocab(counter, vectors=pretrained_vec)
    return v


def split_train_val_test(all_data_path: str) -> tuple:
    # train 0.8, val 0.1, test 0.1
    with open(all_data_path, 'rb') as fopen:
        all_graphs = pickle.load(fopen)
    lens = len(all_graphs)
    train_cnt = round(0.8 * lens)
    val_cnt = round(0.1 * lens)
    # test_cnt = lens - train_cnt - val_cnt
    random.shuffle(all_graphs)
    train_graphs = all_graphs[:train_cnt]
    val_graphs = all_graphs[-2*val_cnt:-val_cnt]
    test_graphs = all_graphs[-val_cnt:]
    # print('all_len=%d, train_len=%d, val_len=%d, test_len=%d' %
    #       (lens, len(train_graphs), len(val_graphs), len(test_graphs)))
    return train_graphs, val_graphs, test_graphs


def prepare_ingredients(pickle_path: str, corpus_type: str,
                        pretrain_name: str = 'glove.840B.300d.txt',
                        emb_cache: str = None,
                        max_vectors: int = 160000,
                        yelp_senti_feature: bool = False,
                        vocab_sos_eos_flag: bool = False,
                        self_loop_flag: bool = True) -> tuple:
    train_graphs, val_graphs, test_graphs = split_train_val_test(pickle_path)
    # glove_pretrained = torchtext.vocab.GloVe(name=pretrain_name, dim=dim,
    #                                          max_vectors=max_vectors)
    emb_pretrained = torchtext.vocab.Vectors(name=pretrain_name, cache=emb_cache,
                                             max_vectors=max_vectors)
    vocab = get_vocab(train_graphs, val_graphs, emb_pretrained, vocab_sos_eos_flag)
    # print("For %s, Vocab size=%d" % (pickle_path, len(vocab)))
    train_set = DocClassificationDataset(train_graphs, vocab, corpus_type=corpus_type,
                                         yelp_senti_feature=yelp_senti_feature,
                                         self_loop_flag=self_loop_flag)
    val_set = DocClassificationDataset(val_graphs, vocab, corpus_type=corpus_type,
                                       yelp_senti_feature=yelp_senti_feature,
                                       self_loop_flag=self_loop_flag)
    test_set = DocClassificationDataset(test_graphs, vocab, corpus_type=corpus_type,
                                        yelp_senti_feature=yelp_senti_feature,
                                        self_loop_flag=self_loop_flag)
    return train_set, val_set, test_set, vocab


if __name__ == '__main__':
    dblp_pickle_path = 'data/dblp.win5.pickle.gz'
    # prepare_ingredients(dblp_pickle_path)
    nyt_pickle_path = 'data/nyt.win5.pickle.gz'
    random.seed(27)
    train_set, val_set, test_set, vocab = prepare_ingredients(nyt_pickle_path, 'nyt')
    print(test_set.docids[:20])
