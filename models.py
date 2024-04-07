import logging
import numpy as np
import torch
import torch.nn as nn
import math
from collections import defaultdict
#t-SNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

from utils.utils import *
from tgn_modules.embedding_module import get_embedding_module

from model.time_encoding import TimeEncode
from gnn.models import CompGCN
from gnn.aggregators import *


class PECF(nn.Module):
    def __init__(self, input_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1,n_layers=2,t_out=1,
                 rnn_layers=1,k=3,method='kmeans',num_s_rel=None,disc_func='lin',alpha=0,beta=0,text_emd_dim=None):
        super().__init__()
        self.h_dim = input_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # initialize rel and ent embedding
        self.node_embeds = nn.Parameter(torch.Tensor(num_ents, input_dim))
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, input_dim))
        self.init_weights()

        self.global_emb = None  
        self.ent_map = None
        self.rel_map = None
        self.graph_dict = None
        self.aggregator= aggregator_event_PECF(input_dim, dropout, num_ents, num_rels, seq_len, maxpool, n_layers,rnn_layers,
                                          self.node_embeds,self.rel_embeds,text_emd_dim)
        if use_gru == 1:
            self.encoder = nn.GRU(input_dim, input_dim, batch_first=True)
        elif use_gru == 2:
            self.encoder = nn.LSTM(input_dim, input_dim, batch_first=True)
        else:
            self.encoder = nn.RNN(input_dim, input_dim, batch_first=True)
        
        #binary class
        #find the most important event
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.decoder = nn.Linear(input_dim+1,1)

        self.threshold = 0.5
        self.out_func = torch.nn.Sigmoid()
        self.criterion = nn.BCELoss()

        #counterfactual
        self.k = k
        self.method = method
        self.target_rel = None
        self.rel_dict = None
        self.num_s_rel = num_s_rel
        self.disc_func = disc_func
        self.alpha = alpha
        self.beta = beta

    def init_weights(self):
        for p in self.parameters():
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
        else:
            stdv = 1. / math.sqrt(p.size(0))
            p.data.uniform_(-stdv, stdv)
    
    def forward(self, t_list, true_prob_r):
        #remove zero
        nonzero_idx = torch.nonzero(t_list, as_tuple=False).view(-1)
        t_list = t_list[nonzero_idx]
        true_prob_r = true_prob_r[nonzero_idx]
        #get history relation data
        rel_data = get_rel_data_list(self.rel_dict,t_list,true_prob_r)

        loss, _ , _= self.__get_pred_loss(t_list,rel_data,true_prob_r)
        return loss
    
    def __get_pred_loss(self, t_list, rel_data, y_true):
        #sorted_t, idx = t_list.sort(0, descending=True)  
        batch_relation_feature = self.aggregator(t_list, self.graph_dict)
        batch_adj = self.__get_edge_adj(t_list)
        #get relation embeds
        #(batch_size,num_rels,dim)
        feature = batch_relation_feature.cuda()
        #get counterfactual enhancement
        embeds_F,embeds_CF,A_F,A_CF = self.__get_embeds(feature,y_true,rel_data,batch_adj)

        pred_F_logit,X_F = self.__get_logit(embeds_F)
        pred_F = self.out_func(pred_F_logit)
        #pred_F = torch.argmax(pred_F_logit,dim=1)

        pred_CF,X_CF = self.__get_logit(embeds_CF)
        pred_CF = self.out_func(pred_CF)
        #get loss
        loss_F = self.criterion(pred_F,A_F)
        loss_CF = self.criterion(pred_CF,A_CF)

        loss_disc = self.__get_disc_loss(embeds_F,embeds_CF)
        loss = loss_F + self.alpha * loss_CF + self.beta * loss_disc

        return loss, pred_F, X_F
    
    def predict(self, t_list, y_data): 
        rel_data = get_rel_data_list(self.rel_dict,t_list,y_data)

        loss, pred_y, X_F = self.__get_pred_loss(t_list,rel_data,y_data)
        return loss, pred_y, X_F

    def evaluate(self, t, y_true):
        loss, pred = self.predict(t, y_true)
        prob_rel = torch.where(pred > self.threshold, 1, 0)
        # target
        y_true = y_true.view(-1)
        return y_true, prob_rel ,loss

    def __get_embeds(self,rel_embeds,A_F,rel_data,batch_adj):
        batch_embeds_F = []
        batch_embeds_CF = []
        A_CF = []
        #batch_size,feature
        for i in range(len(rel_embeds)):
            #(rel_nums,embeds)
            embeds = rel_embeds[i]
            adj = batch_adj[i]
            #every batch
            T,T_cf,all_rel_CF = get_counterfactual(adj,embeds,self.method,self.k,self.target_rel)
            rel_CF = [all_rel_CF[rel_index] for rel_index in rel_data[i]]
            rel_CF = list(set(rel_CF))
            A_F_item = A_F[i].cpu().detach().item()
            A_CF_item = get_A_CF_item(self.rel_dict,rel_CF,A_F_item)
            A_CF.append(A_CF_item)
           
            embeds_final_F = torch.cat((embeds,torch.from_numpy(T).cuda()),axis=1)
            embeds_final_CF = torch.cat((embeds,torch.from_numpy(T_cf).cuda()),axis=1)
            batch_embeds_F.append(embeds_final_F)
            batch_embeds_CF.append(embeds_final_CF)
        
        batch_embeds_F = torch.stack(batch_embeds_F).cuda()
        batch_embeds_CF = torch.stack(batch_embeds_CF).cuda()
        
        A_CF = torch.Tensor(A_CF).cuda()

        return batch_embeds_F.to(torch.float32),batch_embeds_CF.to(torch.float32),A_F,A_CF
    
    def __get_logit(self,embeds):
        #(batch_size,num_rels,dim)->(batch_size,1,dim)
        embeds_pooling = self.maxpooling(embeds).squeeze(1)
        #(batch_size,dim)->(batch_size,2)
        pred_logit = self.decoder(embeds_pooling).squeeze(1)
        #(batch_size)
        return pred_logit,embeds_pooling
    
    def __get_disc_loss(self,embeds_F,embeds_CF):
        sample_F,sample_CF = sample_relation(self.num_s_rel,embeds_F,embeds_CF)
        loss = calc_disc(self.disc_func,sample_F,sample_CF)
        return loss
    
    def __get_edge_adj(self,t_list):
        graph_dict = self.graph_dict
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_list = []
        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
            else:
                time_list.append(torch.LongTensor(times[:length]))
        #each batch get graph
        #(batch_size,rel_num,self.num_ents)
        all_adj_matrices = []
        for index in range(len(t_list)):
            #init adj each rel get adj
            adj_matrices = []
            for _ in range(self.num_rels):
                #adj_matrices.append(np.zeros(self.num_ents))
                adj_matrices.append(np.zeros(self.num_ents))
            g_list = [graph_dict[tim.item()] for tim in time_list[index]]
            g = dgl.batch(g_list)
            rel_type = g.edata['rel_type']
            batch_g_uniq_rel = torch.unique(rel_type, sorted=True)
            for edge_id in batch_g_uniq_rel:
                selected_edges = (g.edata['rel_type'] == edge_id.item()).nonzero().squeeze(1).tolist()
                src_nodes_array,dst_nodes_array = g.edges()
                src_nodes = src_nodes_array[selected_edges].tolist()
                dst_nodes = dst_nodes_array[selected_edges].tolist()
                nodes = list(set(src_nodes+dst_nodes))
                adj_matrices[edge_id][nodes] += 1
            #each edge feature
            all_adj_matrices.append(adj_matrices)
        return np.array(all_adj_matrices)
    
