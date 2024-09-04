import math
import logging
import time
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from model.models import *
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder, soft_cross_entropy
from utils.data_processing import get_data, compute_time_statistics
from utils.event_data_processing import *
from utils.event_data_processing_ml import generate_data_ml
from utils.eval import print_eval_metrics

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
  ### Argument and global variables
  parser = argparse.ArgumentParser('SDM 2025')
  parser.add_argument('--dp', '--data path', type=str, help='Dataset path', default='/data')
  parser.add_argument('--c', '--country', type=str, help='country name', default='IR')

  parser.add_argument('-lt', '--leadtime', type=int, help='lead time', default=1)
  parser.add_argument('-pw', '--predwind', type=int, help='pred wind', default=1)
  parser.add_argument('-hw', '--hist_wind', type=int, help='hist window', default=7)
  parser.add_argument('--bs', type=int, default=1, help='Batch_size')
  parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
  parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
  parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
  parser.add_argument('--n_epoch', type=int, default=1, help='Number of epochs')
  parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')

  parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
  
  parser.add_argument('--patience', type=int, default=1, help='Patience for early stopping')
  parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
  parser.add_argument('--drop_out', type=float, default=0.2, help='Dropout probability')
  parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
  
  parser.add_argument('--dim', type=int, default=128, help='Dimensions of the node embedding')
  #128 64 32 glean,tGCN->2
  parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to'
                                                                    'backprop')
  parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
  parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
  parser.add_argument('--cache_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of cache updater')
  parser.add_argument('--aggregator', type=str, default="mean", help='Type of message '
                                                                          'aggregator')
  parser.add_argument('--max_pool', type=str, default=True)
  parser.add_argument('--message_dim', type=int, default=32, help='Dimensions of the messages')
  parser.add_argument('--cache_dim', type=int, default=32, help='Dimensions of the cache for '
                                                                  'each user')
  parser.add_argument('--uniform', action='store_true',
                      help='take uniform sampling from temporal neighbors')
  parser.add_argument("--use_gru", type=int, default=1, help='1 use gru 0 rnn')
  parser.add_argument("--attn", type=str, default='', help='dot/add/genera; default general')
  parser.add_argument("--rnn-layers", type=int, default=1)

  parser.add_argument("-m","--model", type=str, default="MTG", help="model name")

  parser.add_argument("--k", type=int, default=5, help='number of clusters')
  parser.add_argument("--method", type=str, default='kmeans', help='kmeans,hierarchy,GMM')
  parser.add_argument("--num_s_rels", type=int, default=100, help='number of sample relations')
  parser.add_argument("--disc_func", type=str, default='lin', help='the type of disc_func')
  parser.add_argument("--alpha", type=int, default=0.1, help='loss alpha')
  parser.add_argument("--beta", type=int, default=0.01, help='loss beta')
  parser.add_argument("--text_emd_dim", type=int, default=768, help='text embedding dim')

  parser.add_argument("--agg_mode", type=str, default="GCN", help='agg_mode GCN,SAGEConv,JKNet')
  #secoGD
  parser.add_argument("--encoder", type=str, default="rgcn",help="method of encoder: rgcn/ compgcn")
  parser.add_argument("--decoder", type=str, default="Linear",help="method of decoder")
  # configuration for cross-context hypergraph
  parser.add_argument("--hypergraph_ent", action='store_true', default=False,
                      help="add hypergraph between disentangled nodes")
  parser.add_argument("--hypergraph_rel", action='store_true', default=False,
                      help="add hypergraph between disentangled relations")
  parser.add_argument("--n_layers_hypergraph_ent", type=int, default=1,
                      help="number of propagation rounds on entity hypergraph")
  parser.add_argument("--n_layers_hypergraph_rel", type=int, default=1,
                      help="number of propagation rounds on relation hypergraph")
  parser.add_argument("--score_aggregation", type=str, default='hard',
                      help="score aggregation strategy: hard/ avg")
  parser.add_argument("--k_contexts", type=int, default=5,
                        help="number of contexts to disentangle the sub-embeddings")
  parser.add_argument("--n_hidden", type=int, default=100,
                        help="number of hidden units")
  parser.add_argument("--n_bases", type=int, default=100,
                        help="number of weight blocks for each relation")
  parser.add_argument("--self_loop", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
  parser.add_argument("--layer_norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
  
  try:
    args = parser.parse_args()
  except:
    parser.print_help()
    sys.exit(0)
  print(parser)

  BATCH_SIZE = args.bs
  LEAD_TIME = args.leadtime
  NUM_NEIGHBORS = args.n_degree
  NUM_NEG = 1
  NUM_EPOCH = args.n_epoch
  NUM_HEADS = args.n_head
  DROP_OUT = args.drop_out
  GPU = args.gpu
  DATA = args.c
  NUM_LAYER = args.n_layer
  LEARNING_RATE = args.lr
  NODE_DIM = args.dim
  TIME_DIM = args.dim
  # USE_MEMORY = True
  MESSAGE_DIM = args.message_dim
  MEMORY_DIM = args.cache_dim

  Path("./saved_models/").mkdir(parents=True, exist_ok=True)
  Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
  MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.dp}.pth'
  get_checkpoint_path = lambda \
      epoch: f'./saved_checkpoints/{args.prefix}-{args.dp}-{epoch}.pth'

  ### set up logger
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  Path("log/").mkdir(parents=True, exist_ok=True)
  fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
  fh.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.WARN)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  fh.setFormatter(formatter)
  ch.setFormatter(formatter)
  logger.addHandler(fh)
  logger.addHandler(ch)
  logger.info(args)

  ### Extract data for training, validation and testing
  feature_size = args.dim
  with open(f'{args.dp}/{args.c}/stat.txt', 'r') as fr:
      for line in fr:
        line_split = line.split()
      num_nodes, num_rels = int(line_split[0]), int(line_split[1])

  target_rels = json.load(open(f'{args.dp}/{args.c}/targetRelIds.json'))
  
  x_data, y_data = generate_data(f'{args.dp}/{args.c}/', 'train.txt', 'valid.txt', 'test.txt', target_rels)
  
  #use lead time and predwind
  train_data_x_list, train_data_y_list = \
  divide_data_online(x_data, y_data, lead_time = args.leadtime, pred_wind = args.predwind)
  # cuts to split each set of data to training, validation and test set

  cuts.append([0, 412, 464, 516])
 

  n_sets = len(cuts)
  #add poisson noise
  # noise_intensity = 0
  # poisson_noise = np.random.poisson(noise_intensity, size=2584)

  all_sources, all_destinations, all_timestamps = generate_all(f'{args.dp}/{args.c}/', 'train.txt', 'valid.txt', 'test.txt')

  full_ngh_finder = get_neighbor_finder(f'{args.dp}/{args.c}/', 'train.txt', 'valid.txt', 'test.txt', args.uniform, num_nodes)

  # Set device
  device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
  device = torch.device(device_string)

  # Compute time statistics
  mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(all_sources, all_destinations, all_timestamps)

  with open(args.dp + '/'+ args.c+'/dg_dict.txt', 'rb') as f:
        graph_dict = pickle.load(f)
        print('load dg_dict.txt')

  recall_list  = []
  precision_list = []
  f1_list  = []
  f2_list  = []
  bac_list  = []
  hloss_list = []
  acc_list = []
  auc_list = []

  if args.model == 'glean':
    model = glean_model(h_dim=feature_size, num_ents=num_nodes,
                            num_rels=num_rels, dropout=DROP_OUT, 
                            seq_len=args.hist_wind,
                            maxpool=args.max_pool,
                            use_gru=args.use_gru,
                            attn=args.attn)
    with open(args.dp + '/'+ args.c+'/wg_dict_truncated.txt', 'rb') as f:
        word_graph_dict = pickle.load(f)
        print('load wg_dict_truncated.txt')
    with open(args.dp+ '/'+ args.c+'/word_relation_map.txt', 'rb') as f:
        rel_map = pickle.load(f)
        print('load word_relation_map.txt')
    with open(args.dp + '/'+ args.c+'/word_entity_map.txt', 'rb') as f:
        ent_map = pickle.load(f)
        print('load word_entity_map.txt')
    with open('{}/{}/768.w_emb'.format(args.dp, args.c), 'rb') as f:
          word_embeds = pickle.load(f,encoding="latin1")
          word_embeds = torch.FloatTensor(word_embeds)
          vocab_size = word_embeds.size(0)
    
    model.word_embeds = word_embeds
    model.graph_dict = graph_dict
    model.word_graph_dict = word_graph_dict
    model.ent_map = ent_map
    model.rel_map = rel_map

  elif args.model == 'MTG':
    model = MTG_model(neighbor_finder = full_ngh_finder, dim=feature_size,
              num_nodes=num_nodes,
              num_edges=num_rels, device=device, max_pool = args.max_pool,
              graph_dict = graph_dict, hist_wind = args.hist_wind,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_cache=True,
              message_dimension=MESSAGE_DIM, cache_dimension=MEMORY_DIM,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              cache_updater_type=args.cache_updater,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst)
  
  elif args.model == 'PECF':
    model = PECF(input_dim=feature_size, num_ents=num_nodes,
                            num_rels=num_rels, dropout=DROP_OUT,
                            seq_len=args.hist_wind,
                            maxpool=args.max_pool,
                            use_gru=args.use_gru,
                            n_layers = NUM_LAYER,
                            rnn_layers=args.rnn_layers,
                            k=args.k,
                            method=args.method,
                            num_s_rel=args.num_s_rels,
                            disc_func=args.disc_func,
                            alpha=args.alpha,beta=args.beta,
                            text_emd_dim=args.text_emd_dim)
    with open(args.dp + '/' + args.c+'/rel_dict.txt', 'rb') as f:
        rel_dict = pickle.load(f)
    print('load rel_dict.txt')
    
    model.graph_dict = graph_dict
    model.target_rel = target_rels
    model.rel_dict = rel_dict
  
  elif args.model == "CompGCN+RNN":
    model = CompGCN_RNN_model(h_dim=feature_size, num_ents=num_nodes,
                            num_rels=num_rels, dropout=DROP_OUT, 
                            seq_len=args.hist_wind,
                            maxpool=args.max_pool,
                            use_gru=args.use_gru)
    model.graph_dict = graph_dict

  elif args.model == "TGN":
     model = TGN(neighbor_finder=full_ngh_finder, dim=feature_size,
              num_nodes=num_nodes,num_edges=num_rels, device=device,n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT,
            message_dimension=feature_size, memory_dimension=feature_size,n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst)
     
  elif args.model == "DynamicGCN":
    with open(args.dp + '/'+ args.c+'/wg_dict_truncated.txt', 'rb') as f:
        word_graph_dict = pickle.load(f)
        print('load wg_dict_truncated.txt')
    with open('{}/{}/768.w_emb'.format(args.dp, args.c), 'rb') as f:
          word_embeds = pickle.load(f,encoding="latin1")
          word_embeds = torch.FloatTensor(word_embeds)
          vocab_size = word_embeds.size(0)
          
    model = DynamicGCN(h_dim=feature_size, num_ents=num_nodes,
                            num_rels=num_rels, dropout=DROP_OUT, 
                            seq_len=args.hist_wind,
                            vocab_size=vocab_size)
    
    model.word_embeds = word_embeds.to(device)
    model.word_graph_dict = word_graph_dict

  elif args.model == "TGCN":
    with open(args.dp + '/'+ args.c+'/wg_dict_truncated.txt', 'rb') as f:
        word_graph_dict = pickle.load(f)
        print('load wg_dict_truncated.txt')
    with open('{}/{}/768.w_emb'.format(args.dp, args.c), 'rb') as f:
          word_embeds = pickle.load(f,encoding="latin1")
          word_embeds = torch.FloatTensor(word_embeds)
          vocab_size = word_embeds.size(0)
          
    model = TGCN(h_dim=feature_size, num_ents=num_nodes,
                            num_rels=num_rels, dropout=DROP_OUT, 
                            seq_len=args.hist_wind,
                            vocab_size=vocab_size)
    
    model.word_embeds = word_embeds.to(device)
    model.word_graph_dict = word_graph_dict
  
  elif args.model == "tRGCN":
    model = RGCN_RNN_model(h_dim=feature_size, num_ents=num_nodes,
                            num_rels=num_rels, dropout=DROP_OUT, 
                            seq_len=args.hist_wind,
                            maxpool=args.max_pool,
                            use_gru=args.use_gru)
    model.graph_dict = graph_dict
  
  elif args.model == 'CFLP':
    model = CFLP(input_dim=feature_size, num_ents=num_nodes,
                            num_rels=num_rels, dropout=DROP_OUT,
                            seq_len=args.hist_wind,
                            maxpool=args.max_pool,
                            use_gru=args.use_gru,
                            n_layers = NUM_LAYER,
                            rnn_layers=args.rnn_layers,
                            k=args.k,
                            method=args.method,
                            num_s_rel=args.num_s_rels,
                            disc_func=args.disc_func,
                            alpha=args.alpha,beta=args.beta,
                            agg_mode=args.agg_mode)
    with open(args.dp + '/' + args.c+'/rel_dict.txt', 'rb') as f:
        rel_dict = pickle.load(f)
    print('load rel_dict.txt')
    
    model.graph_dict = graph_dict
    model.target_rel = target_rels
    model.rel_dict = rel_dict

  elif args.model == 'SeCoGD':
     with open(os.path.join('/data_disentangled/' +args.c+"_LDA_K5"+ '/graph_dict_each_context.pkl'), 'rb') as fp:
        graph_dict = pickle.load(fp)
     hyper_adj_ent = torch.load("/data_disentangled/"+args.c+"_LDA_K5"+"/hypergraph_ent.pt")
     hyper_adj_rel = torch.load("/data_disentangled/"+args.c+"_LDA_K5"+"/hypergraph_rel.pt")

     model = SeCo(args.decoder,
                 args.encoder,
                 num_nodes,
                 num_rels,
                 hyper_adj_ent,
                 hyper_adj_rel,
                 args.n_layers_hypergraph_ent,
                 args.n_layers_hypergraph_rel,
                 args.k_contexts,
                 args.n_hidden,
                 sequence_len=args.hist_wind,
                 num_bases=args.n_bases,
                 num_hidden_layers=NUM_LAYER,
                 dropout=DROP_OUT,
                 self_loop=args.self_loop,
                 layer_norm=args.layer_norm,
                 input_dropout=DROP_OUT,
                 hidden_dropout=DROP_OUT,
                 feat_dropout=DROP_OUT,
                 use_cuda=True,
                 gpu=args.gpu,
                 graph_dict=graph_dict)
     
  for i in range(args.n_runs):
    results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)
    y_trues_test_list = []
    y_hats_test_list = []
    for s in range(n_sets):
        
        model_name = model.__class__.__name__
        print('Model:', model_name)
        
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('#params:', total_params)
        model = model.to(device)
        all_data_x, all_data_y = train_data_x_list[s], train_data_y_list[s]
        cut_1, cut_2, cut_3, cut_4 = cuts[s]
        logger.info('num of training instances: {}'.format(cut_2))

        best_hloss = float('inf')
        no_improvement = 0
        early_stopper = EarlyStopMonitor(max_round=args.patience)
        for epoch in range(NUM_EPOCH):
          ### Training
          if args.model == 'MTG':
            # Reinitialize cache and memory of the model at the start of each epoch
            model.entity_cache.__init_cache__()
            model.rel_cache.__init_cache__()
            model.memory.__init_memory__()
            # Train using only training graph
            model.set_neighbor_finder(full_ngh_finder)
          elif args.model == 'TGN':
            model.memory.__init_memory__()

          m_loss = []
          logger.info('start {} epoch'.format(epoch))
          batch_idx = 1
          cut_1 = 1
          X_embed = []
          y_label = []
          #x:t-h y:label
          for k in tqdm(range(cut_1, cut_2, args.backprop_every)):
            model.train()
            train_loss = 0.
            num_samples = 0
            optimizer.zero_grad()
            # Custom loop to allow to perform backpropagation only every a certain number of batches
            while num_samples < args.backprop_every:
              if batch_idx >= cut_2:
                  break
              sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, story_ids_batch = \
              get_batch_data(all_data_x[batch_idx],poisson_noise[batch_idx])
              #time
              t_h = batch_idx
              if t_h not in graph_dict:
                  batch_idx += 1
                  continue
              num_samples += 1
              y_true = torch.tensor([all_data_y[t_h]], requires_grad=False).float().to(device)
              if args.model == 'MTG':
                y_hat = model.predict(sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, t_h, story_ids_batch)
                loss = criterion(y_hat, y_true)
              elif args.model == 'glean' or args.model == "CompGCN+RNN" or args.model == "DynamicGCN" or args.model == "TGCN" or args.model == "tRGCN" or args.model == 'SeCoGD':
                 y_hat = model.predict([t_h])
                 #print(y_hat)
                 loss = criterion(y_hat, y_true)
              elif args.model == 'PECF' or args.model == 'CFLP':
                 loss, y_hat, embed_F = model.predict([t_h],y_true)
                 X_embed.append(embed_F)
                 y_label.append(y_true)
              elif args.model == 'TGN':
                 y_hat = model.predict(sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
                 loss = criterion(y_hat, y_true)

              train_loss += loss
              batch_idx += 1
            if train_loss > 0.:
                #print("train_loss:",train_loss.item())
                loss.backward(retain_graph=False)
                optimizer.step()
                m_loss.append(loss.item())
            # Detach cache and memory after 'args.backprop_every' number of batches so we don't backpropagate to
            # the start of time
            if args.model == 'MTG':
              model.entity_cache.detach_cache()
              model.rel_cache.detach_cache()
              model.memory.detach_memory()
            elif args.model == 'TGN':
              model.memory.detach_memory()
          
          # validation
          valid_loss = 0.
          y_hats_valid = []
          y_trues_valid = []
          for k in range(cut_2, cut_3):
              model.eval()
              with torch.no_grad():
                sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, story_ids_batch = \
                get_batch_data(all_data_x[k],0)
                t_h = k
                if t_h not in graph_dict:
                    continue
                y_true = torch.tensor([all_data_y[t_h]], requires_grad=False).float().to(device)
                if args.model == 'MTG':
                  y_hat_logits = model.predict(sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, t_h, story_ids_batch)
                elif args.model == 'glean' or args.model == "CompGCN+RNN" or args.model == "DynamicGCN" or args.model == "TGCN" or args.model == "tRGCN" or args.model == 'SeCoGD':
                  y_hat_logits = model.predict([t_h])
                  #print(y_hat_logits)
                elif args.model == 'PECF' or args.model == 'CFLP':
                  _, y_hat_logits,_ = model.predict([t_h],y_true)
                elif args.model == 'TGN':
                  y_hat_logits = model.predict(sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
                
                #y_hat = torch.argmax(y_hat_logits, dim=1)
                valid_loss += criterion(y_hat_logits, y_true)
                y_hats_valid.append(y_hat_logits.item())
                y_trues_valid.append(y_true.item())

          if args.model == 'MTG':
              model.entity_cache.detach_cache()
              model.rel_cache.detach_cache()
              model.memory.detach_memory()
          elif args.model == 'TGN':
              model.memory.detach_memory()
          hloss, recall, precision ,f1, f2, bac, acc = print_eval_metrics(y_trues_valid, y_hats_valid)
          #valid loss
          if valid_loss < best_hloss:
            no_improvement = 0
            best_hloss = valid_loss
            test_loss = 0.
            y_hats_test = []
            y_trues_test = []
            for k in range(cut_3, cut_4):
              model.eval()
              with torch.no_grad():
                sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, story_ids_batch = \
                get_batch_data(all_data_x[k],0)
                t_h = k
                if t_h not in graph_dict:
                    continue
                y_true = torch.tensor([all_data_y[t_h]], requires_grad=False).float().to(device)
                #y_hat = model.predict(sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
                if args.model == 'MTG':
                  y_hat_logits = model.predict(sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, t_h, story_ids_batch)
                elif args.model == 'glean' or args.model =="CompGCN+RNN" or args.model == "DynamicGCN" or args.model == "TGCN" or args.model == "tRGCN" or args.model == 'SeCoGD':
                  y_hat_logits = model.predict([t_h])
                elif args.model == 'PECF' or args.model == 'CFLP':
                  _, y_hat_logits, _ = model.predict([t_h],y_true)
                elif args.model == 'TGN':
                  y_hat_logits = model.predict(sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
                
                #y_hat = torch.argmax(y_hat_logits, dim=1)
                test_loss += criterion(y_hat_logits, y_true)
                y_hats_test.append(y_hat_logits.item())
                y_trues_test.append(y_true.item())

            _,_,_,_,_,_,_ = print_eval_metrics(y_trues_test, y_hats_test)
          else:
            no_improvement += 1
            print ('no_improvement:', no_improvement)

          if no_improvement > args.patience:
            logger.info('No improvement over {} epochs, stop training'.format(args.patience))
            break

          if args.model == 'MTG':
              model.entity_cache.detach_cache()
              model.rel_cache.detach_cache()
              model.memory.detach_memory()
          elif args.model == 'TGN':
              model.memory.detach_memory()
        #clear gpu
        torch.cuda.empty_cache()

        y_trues_test_list.extend(y_trues_test)
        y_hats_test_list.extend(y_hats_test)

    test_hloss, test_recall, test_precision, test_f1, test_f2, test_bac, test_acc = \
    print_eval_metrics(y_trues_test_list, y_hats_test_list)
    recall_list.append(test_recall)
    precision_list.append(test_precision)
    f1_list.append(test_f1)
    f2_list.append(test_f2)
    bac_list.append(test_bac)
    acc_list.append(test_acc)
    hloss_list.append(test_hloss)

  print('finish training, results ....')
  # save average results
  recall_list = np.array(recall_list)
  precision_list = np.array(precision_list)
  f1_list = np.array(f1_list)
  f2_list = np.array(f2_list)
  bac_list = np.array(bac_list)
  hloss_list = np.array(hloss_list)
  acc_list = np.array(acc_list)
  
  recall_avg, recall_std = recall_list.mean(0), recall_list.std(0)
  precision_avg, precision_std = precision_list.mean(0), precision_list.std(0)
  f1_avg, f1_std = f1_list.mean(0), f1_list.std(0)
  f2_avg, f2_std = f2_list.mean(0), f2_list.std(0)
  bac_avg, bac_std = bac_list.mean(0), bac_list.std(0)
  acc_avg, acc_std = acc_list.mean(0), acc_list.std(0)
  hloss_avg, hloss_std = hloss_list.mean(0), hloss_list.std(0)

  print('--------------------')

  print("Rec  weighted: {:.4f}".format(recall_avg))
  print("Precision  weighted: {:.4f}".format(precision_avg))
  print("F1   weighted: {:.4f}".format(f1_avg))
  beta=2
  print("F2   weighted: {:.4f}".format(f2_avg))
  print("BAC  weighted: {:.4f}".format(bac_avg))
  print("Accuracy  weighted: {:.4f}".format(acc_avg))
  print("Loss: {:.4f}".format(hloss_avg))