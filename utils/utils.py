import numpy as np
import torch
import torch.nn as nn
import numpy as np
import os
from math import log
import scipy.sparse as sp

import dgl
import torch
import pickle
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, recall_score, precision_score, fbeta_score, hamming_loss, zero_one_loss
from sklearn.metrics import jaccard_score
from sknetwork.embedding import Spectral
from sknetwork.utils import get_membership
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans,AgglomerativeClustering
import torch.nn.functional as F

import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import inv, eigs
import networkx as nx
from multiprocessing import Pool

def node_norm_to_edge_norm(G):
    G = G.local_var()
    G.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    
def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

 
def get_data_with_t(data, time):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == time]
    return np.array(triples)


def get_data_idx_with_t_r(data, t,r):
    for i, quad in enumerate(data):
        if quad[3] == t and quad[1] == r:
            return i
    return None

 
def load_quadruples(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            story_id = int(line_split[4])
            quadrupleList.append([head, rel, tail, time, story_id])
            times.add(time)
    #get times
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)
 
'''
Customized collate function for Pytorch data loader
'''
 
def collate_2(batch):
    batch_data = [item[0] for item in batch]
    out_target = [item[1] for item in batch]
    data = [item[2] for item in batch]
    return [batch_data, out_target, data]

def collate_4(batch):
    batch_data = [item[0] for item in batch]
    s_prob = [item[1] for item in batch]
    r_prob = [item[2] for item in batch]
    o_prob = [item[3] for item in batch]
    return [batch_data, s_prob, r_prob, o_prob]

def collate_6(batch):
    inp0 = [item[0] for item in batch]
    inp1 = [item[1] for item in batch]
    inp2 = [item[2] for item in batch]
    inp3 = [item[3] for item in batch]
    inp4 = [item[4] for item in batch]
    inp5 = [item[5] for item in batch]
    return [inp0, inp1, inp2, inp3, inp4, inp5]


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor

def move_dgl_to_cuda(g):
    if torch.cuda.is_available():
        g.ndata.update({k: cuda(g.ndata[k]) for k in g.ndata})
        g.edata.update({k: cuda(g.edata[k]) for k in g.edata})

 
'''
Get sorted r to make batch for RNN (sorted by length)
'''
def get_sorted_r_t_graphs(t, r, r_hist, r_hist_t, graph_dict, word_graph_dict, reverse=False):
    r_hist_len = torch.LongTensor(list(map(len, r_hist)))
    if torch.cuda.is_available():
        r_hist_len = r_hist_len.cuda()
    r_len, idx = r_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(r_len,as_tuple=False))
    r_len_non_zero = r_len[:num_non_zero]
    idx_non_zero = idx[:num_non_zero]  
    idx_zero = idx[num_non_zero-1:]  
    if torch.max(r_hist_len) == 0:
        return None, None, r_len_non_zero, [], idx, num_non_zero
    r_sorted = r[idx]
    r_hist_t_sorted = [r_hist_t[i] for i in idx]
    g_list = []
    wg_list = []
    r_ids_graph = []
    r_ids = 0 # first edge is r 
    for t_i in range(len(r_hist_t_sorted[:num_non_zero])):
        for tim in r_hist_t_sorted[t_i]:
            try:
                wg_list.append(word_graph_dict[r_sorted[t_i].item()][tim])
            except:
                pass

            try:
                sub_g = graph_dict[r_sorted[t_i].item()][tim]
                if sub_g is not None:
                    g_list.append(sub_g)
                    r_ids_graph.append(r_ids) 
                    r_ids += sub_g.number_of_edges()
            except:
                continue
    if len(wg_list) > 0:
        batched_wg = dgl.batch(wg_list)
    else:
        batched_wg = None
    if len(g_list) > 0:
        batched_g = dgl.batch(g_list)
    else:
        batched_g = None
    
    return batched_g, batched_wg, r_len_non_zero, r_ids_graph, idx, num_non_zero
 
 

'''
Loss function
'''
# Pick-all-labels normalised (PAL-N)
def soft_cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=-1) # pred (batch, #node/#rel)
    pred = pred.type('torch.DoubleTensor')
    if torch.cuda.is_available():
        pred = pred.cuda()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

 
'''
Generate/get (r,t,s_count, o_count) datasets 
'''
def get_scaled_tr_dataset(num_nodes, path='../data/', dataset='india', set_name='train', seq_len=7, num_r=None):
    import pandas as pd
    from scipy import sparse
    file_path = '{}{}/tr_data_{}_sl{}_rand_{}.npy'.format(path, dataset, set_name, seq_len, num_r)
    if not os.path.exists(file_path):
        print(file_path,'not exists STOP for now')
        exit()
    else:
        print('load tr_data ...',dataset,set_name)
        with open(file_path, 'rb') as f:
            [t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o] = pickle.load(f)
    t_data = torch.from_numpy(t_data)
    r_data = torch.from_numpy(r_data)
    true_prob_s = torch.from_numpy(true_prob_s.toarray())
    true_prob_o = torch.from_numpy(true_prob_o.toarray())
    return t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o
 
'''
Empirical distribution
'''
def get_true_distributions(path, data, num_nodes, num_rels, dataset='india', set_name='train'):
    """ (# of s-related triples) / (total # of triples) """
     
    file_path = '{}{}/true_probs_{}.npy'.format(path, dataset, set_name)
    if not os.path.exists(file_path):
        print('build true distributions...',dataset,set_name)
        time_l = list(set(data[:,-2]))
        time_l = sorted(time_l,reverse=False)
        true_prob_s = None
        true_prob_o = None
        true_prob_r = None
        for cur_t in time_l:
            triples = get_data_with_t(data, cur_t)
            true_s = np.zeros(num_nodes)
            true_o = np.zeros(num_nodes)
            true_r = np.zeros(num_rels)
            s_arr = triples[:,0]
            o_arr = triples[:,2]
            r_arr = triples[:,1]
            for s in s_arr:
                true_s[s] += 1
            for o in o_arr:
                true_o[o] += 1
            for r in r_arr:
                true_r[r] += 1
            true_s = true_s / np.sum(true_s)
            true_o = true_o / np.sum(true_o)
            true_r = true_r / np.sum(true_r)
            if true_prob_s is None:
                true_prob_s = true_s.reshape(1, num_nodes)
                true_prob_o = true_o.reshape(1, num_nodes)
                true_prob_r = true_r.reshape(1, num_rels)
            else:
                true_prob_s = np.concatenate((true_prob_s, true_s.reshape(1, num_nodes)), axis=0)
                true_prob_o = np.concatenate((true_prob_o, true_o.reshape(1, num_nodes)), axis=0)
                true_prob_r = np.concatenate((true_prob_r, true_r.reshape(1, num_rels)), axis=0)
             
        with open(file_path, 'wb') as fp:
            pickle.dump([true_prob_s,true_prob_r,true_prob_o], fp)
    else:
        print('load true distributions...',dataset,set_name)
        with open(file_path, 'rb') as f:
            [true_prob_s, true_prob_r, true_prob_o] = pickle.load(f)
    true_prob_r = torch.from_numpy(true_prob_r)
    return true_prob_r 

 

'''
Evaluation metrics
'''
# Label based
class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)
  
class MergeLayer_tgn(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim2)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x):
    #x = torch.cat([x1, x2], dim=1)
    x = self.fc1(x)
    h = self.act(self.fc2(x))
    return h


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)

def get_neighbor_finder(inPath, fileName, fileName2, fileName3, uniform, num_node):
  adj_list = [[] for _ in range(num_node)]
  with open(os.path.join(inPath, fileName), 'r') as fr:
      for line in fr:
          line_split = line.split()
          head = int(line_split[0])
          tail = int(line_split[2])
          rel = int(line_split[1])-1
          time = int(line_split[3])
          adj_list[head].append((tail, rel, time))
          # adj_list[destination].append((source, edge_idx, timestamp))
  with open(os.path.join(inPath, fileName2), 'r') as fr:
    for line in fr:
        line_split = line.split()
        head = int(line_split[0])
        tail = int(line_split[2])
        rel = int(line_split[1])-1
        time = int(line_split[3])
        adj_list[head].append((tail, rel, time))
        # adj_list[destination].append((source, edge_idx, timestamp))
  with open(os.path.join(inPath, fileName3), 'r') as fr:
    for line in fr:
        line_split = line.split()
        head = int(line_split[0])
        tail = int(line_split[2])
        rel = int(line_split[1])-1
        time = int(line_split[3])
        adj_list[head].append((tail, rel, time))
        # adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)

class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times

def soft_cross_entropy(pred, soft_targets):
    print (pred)
    logsoftmax = torch.nn.LogSoftmax(dim=-1) # pred (batch, #node/#rel)
    print (logsoftmax(pred))
    pred = pred.type('torch.DoubleTensor')
    if torch.cuda.is_available():
        pred = pred.cuda()
    return - soft_targets * logsoftmax(pred)

def get_sentence_embeddings(graph_dict, source_cache, edge_times, device):
    # n = len(story_ids)
    # sentence_embeddings = torch.nn.Parameter(torch.zeros(n, embedding_size), requires_grad=True).to(device)
    # nn.init.xavier_uniform_(sentence_embeddings, gain=nn.init.calculate_gain('relu'))
    n = len(source_cache)
    edge_times = edge_times.tolist()
    time = list(set(edge_times))
    g_list = graph_dict[time[0]]
    sentence_embeddings = g_list.edata['text_emb'].to(device)
    if len(sentence_embeddings) != n:
       print("!!")
       print(time)
       sentence_embeddings = torch.nn.Parameter(torch.zeros(n, 768), requires_grad=True).to(device)
       nn.init.xavier_uniform_(sentence_embeddings, gain=nn.init.calculate_gain('relu'))
    return sentence_embeddings

def Kmeans_clustering(adj, k,target_rel):
    kmeans = KMeans(n_clusters = k,n_init= 'auto').fit(adj)
    rel_class = kmeans.labels_
    T = class_to_T(rel_class,target_rel)
    #(rel,)
    return T

def hierarchy_cluster(adj,k,target_rel):
   model = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(adj)
   rel_class = model.labels_
   T = class_to_T(rel_class,target_rel)
   return T

def GaussianMixture_cluster(adj,k,target_rel):
   model = GaussianMixture(n_components=k).fit(adj)
   rel_class = model.predict(adj)
   T = class_to_T(rel_class,target_rel)
   return T

def class_to_T(rel_class,target_id):
    target_list = [rel_class[i] for i in target_id]
    target_class = max(target_list,key=target_list.count)
    T = []
    for item in rel_class:
        if item == target_class:
            T.append(1)
        else:
            T.append(0)
    return T

def get_t(adj, method, k, target_rel):
    if method == 'kmeans':
        T = Kmeans_clustering(adj, k, target_rel)
    elif method == 'hierarchy':
        T = hierarchy_cluster(adj, k, target_rel)
    elif method == 'GMM':
       T = GaussianMixture_cluster(adj, k, target_rel)
    return np.array(T)

def get_CF(rel_embs, T_f, dist='euclidean', thresh=50):
    if dist == 'euclidean':
        # Euclidean distance
        simi_mat = cdist(rel_embs.cpu().detach().numpy(), rel_embs.cpu().detach().numpy(), 'euclidean')
    thresh = np.percentile(simi_mat, thresh)
    # give selfloop largest distance
    np.fill_diagonal(simi_mat, np.max(simi_mat)+1)
    # nearest neighbor relation index for each relation shape(rel_num,rel_num)
    relation_nns = np.argsort(simi_mat, axis=1)
    # find nearest CF relation for each relation
    relation_iter = list(range(rel_embs.shape[0]))
    results,all_rel_CF = get_CF_single(relation_iter,relation_nns,T_f,thresh,simi_mat)
    T_cf = results.reshape(-1,1)
    return T_cf,all_rel_CF

def get_CF_single(relation_iter,relation_nns,T_f,thresh,simi_mat):
    """ single process for getting CF relation """

    T_cf = np.zeros(len(relation_iter))
    all_rel_CF = []
    for rel in relation_iter:
        # for each relation rel, find the nearest relation
        nns_rel = relation_nns[rel]
        i = 0
        while i < len(nns_rel)-1:
            if simi_mat[rel, nns_rel[i]]  > thresh:
                T_cf[rel] = T_f[rel]
                all_rel_CF.append(rel)
                break
            elif T_f[rel] != T_f[nns_rel[i]]:
                T_cf[rel] = 1 - T_f[rel]  # T_f[nns_a[i], nns_b[j]] when treatment not binary
                all_rel_CF.append(nns_rel[i])
                break
            else:
                i += 1
    return T_cf,all_rel_CF

def get_counterfactual(adj,rel_embeds,method,k,target_rel):
        #get treatment
        T = get_t(adj,method,k,target_rel).reshape(-1,1)
        #get cf
        T_cf,all_rel_CF = get_CF(rel_embeds,T)
        return T,T_cf,all_rel_CF

def get_A_CF_item(rel_dict,rel_CF,A_T_item):
    for k,v in rel_dict.items():
        if v == rel_CF:
            key = k.split('/')
            return float(key[1])
    return A_T_item

def get_rel_data_list(rel_dict,t_list,y_true):
    rel_data = []
    #t_list = t_list.cpu().detach().tolist()
    y_true = y_true.cpu().detach()
    for i in range(len(t_list)):
        rel_key = str(int(t_list[i])+7)+'/'+str(int(y_true[i]))
        rel_data.append(rel_dict[rel_key])
    return rel_data

def sample_relation(num_s_rel, rel_f_feat, rel_cf_feat):
    # TODO: add sampling with separated treatments
    f_idx = np.random.choice(len(rel_f_feat), min(num_s_rel,len(rel_f_feat)), replace=False)
    np_f = rel_f_feat[f_idx]
    cf_idx = np.random.choice(len(rel_cf_feat), min(num_s_rel,len(rel_cf_feat)), replace=False)
    np_cf = rel_cf_feat[cf_idx]
    return np_f, np_cf

def calc_disc(disc_func, rel_s_f_feat, rel_s_cf_feat):
    X_f = rel_s_f_feat
    X_cf = rel_s_cf_feat
    if disc_func == 'lin':
        mean_f = X_f.mean(0)
        mean_cf = X_cf.mean(0)
        loss_disc = torch.sqrt(F.mse_loss(mean_f, mean_cf) + 1e-6)
    elif disc_func == 'kl':
        # TODO: kl divergence
        pass
    else:
        raise Exception('unsupported distance function for discrepancy loss')
    return loss_disc
