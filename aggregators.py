import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from gnn.utils import *
from gnn.propagations import *
from gnn.modules import *
from utils.utils import get_sentence_embeddings
import time
from itertools import groupby
from torch_scatter import scatter
from dgl.nn.pytorch.conv import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_layers < 2:
            self.layers.append(GCNLayer(in_feats, n_classes, activation, dropout))
        else:
            self.layers.append(GCNLayer(in_feats, n_hidden, activation, dropout))
            for i in range(n_layers - 2):
                self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
            self.layers.append(GCNLayer(n_hidden, n_classes, activation, dropout)) # activation or None

    def forward(self, g, features=None): # no reverse
        if features is None:
            h = g.ndata['h']
        else:
            h = features
        for layer in self.layers:
            h = layer(g, h)
        return h
    


class aggregator_event_PECF(nn.Module):
    def __init__(self, node_in_feat, dropout, num_nodes, num_rels, seq_len=7, maxpool=1,n_layers=2,rnn_layers=1,node_embeds=None,rel_embeds=None,text_emb_dim=None):
        super().__init__()
        self.node_in_feat = node_in_feat  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.rnn_layers = rnn_layers
        self.node_embeds = node_embeds
        self.rel_embeds = rel_embeds
        self.text_w = nn.Linear(text_emb_dim,node_in_feat,bias=True)
        self.activate = nn.ReLU()
        self.edge_w = nn.Linear(node_in_feat*3,node_in_feat,bias=True)

        self.encoder = nn.GRU(node_in_feat, node_in_feat, batch_first=True)

        out_feat = int(node_in_feat // 2)
        hid_dim = int(node_in_feat *2)

        self.re_aggr1 = RGCN_dg(node_in_feat, hid_dim,out_feat,node_in_feat,out_feat,n_layers,num_rels,regularizer="basis",num_bases=None,
                                use_bias=True,activation=F.relu, use_self_loop=True,
                                layer_norm=False,low_mem=False, dropout=0.2, text_emb_dim=text_emb_dim)
        self.re_aggr2 = RGCN_dg(out_feat, hid_dim,node_in_feat,out_feat,node_in_feat,n_layers,num_rels,regularizer="basis",num_bases=None,
                                use_bias=True,activation=F.relu, use_self_loop=True,
                                layer_norm=False,low_mem=False, dropout=0.2, text_emb_dim=text_emb_dim)
        # self.re_aggr1 = CompGCN_dg_PECF(node_in_feat, hid_dim, node_in_feat, hid_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        # self.re_aggr2 = CompGCN_dg_PECF(hid_dim, node_in_feat, hid_dim, node_in_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        
        # self.re_aggr1 = GCN_PECF(node_in_feat, hid_dim, node_in_feat, node_in_feat, node_in_feat, n_layers, activation=F.relu) # to be defined
        # self.re_aggr2 = GCN_PECF(node_in_feat, hid_dim, node_in_feat, node_in_feat, node_in_feat, n_layers, activation=F.relu) # to be defined
        
    def __get_rel_embedding(self,batch_g):
        init_rel_embeds = torch.zeros(self.num_rels, self.node_in_feat).cuda()
        #caclulate rel embedding
        batch_g_rel = batch_g.edata['rel_type'].long()
        batch_g_uniq_rel = torch.unique(batch_g_rel, sorted=True)
        # aggregate edge embedding by relation
        batch_edge_emb_avg_by_rel_ = scatter(batch_g.edata['e_h'],batch_g_rel,dim=0,reduce="mean")  # shape=(max rel in batch_G, relaion text emb dim)
        #get uniq rel embedding
        init_rel_embeds[batch_g_uniq_rel] = batch_edge_emb_avg_by_rel_[batch_g_uniq_rel]
        return init_rel_embeds

    def forward(self, t_list, graph_dict):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_list = []
        len_non_zero = []

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)
        
        #init embeds
        init_node_embeds = self.node_embeds
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #each time
        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        #entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        batched_graph = dgl.batch(g_list)
        batched_graph = batched_graph.to(device)   #gpu
        
        #init node embedding
        node_id = batched_graph.ndata['id']
        batched_graph.ndata['h'] = init_node_embeds[node_id].view(-1, init_node_embeds.shape[1]).to(device)
        batched_graph.edata['text_h'] = self.text_w(batched_graph.edata['text_emb'])

        #caclulate node embedding
        self.re_aggr1(batched_graph)
        self.re_aggr2(batched_graph)
        #caclulate edge embedding
        batch_g_src, batch_g_dst = batched_graph.edges()
        batch_g_src_nid = batch_g_src.long()
        batch_g_dst_nid = batch_g_dst.long()

        batch_dyn_node2src_embeds = batched_graph.ndata['h'][batch_g_src_nid].to(device)
        batch_dyn_node2dst_embeds = batched_graph.ndata['h'][batch_g_dst_nid].to(device)
        batch_g_edge_text_embeds = batched_graph.edata['text_h']

        batched_graph.edata['e_h'] = torch.cat([batch_dyn_node2src_embeds,batch_g_edge_text_embeds,
                                                        batch_dyn_node2dst_embeds], dim=-1)
        batched_graph.edata['e_h'] = self.edge_w(batched_graph.edata['e_h'])
        batched_graph.edata['e_h'] = self.activate(batched_graph.edata['e_h'])

        #print("batched_graph.edata['e_h']",batched_graph.edata['e_h'])
        g_list = dgl.unbatch(batched_graph)
        batched_embed_seq_tensor = []
        embed_seq_tensor = []
        for i, times in enumerate(time_list):
            batched_embed_seq_tensor = []
            for j, t in enumerate(times):
               batched_embed_seq_tensor.append(self.__get_rel_embedding(g_list[time_to_idx[t.item()]]))
            batched_embed_seq_tensor = torch.stack(batched_embed_seq_tensor).to(device)
            #(num_rels,seq_len,feature)
            output,hn = self.encoder(batched_embed_seq_tensor.transpose(0, 1))
            #(num_rels,feature)
            rel_embeds = hn.squeeze(0)
            embed_seq_tensor.append(rel_embeds)
        embed_seq_tensor = torch.stack(embed_seq_tensor).to(device)
        

        return embed_seq_tensor
    