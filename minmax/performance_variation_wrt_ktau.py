import argparse
import numpy as np
import random
import time
import sys
import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import  pad_sequence

from common import logger, set_log
from minmax.graphs import PermGnnGraph, LinqsUmdDocumentGraph, TwitterGraph
from minmax.main import PermutationGenerator, PermutationInvariantGNN
from minmax.linkpred_utils import prep_permgnn_graph,  fetch_lp_data_split 
from minmax.utils import cudavar,load_model
from minmax.earlystopping import EarlyStoppingModule
import pickle
from scipy import stats
import matplotlib.pyplot as plt

def generate_global_permutations(av,gr,num_perms):
  """
    Given a dataset and train-val-test split, generate golbal node permutations 
    to check behavior for all three variations - PermGNN, MultiPerm, 1Perm
  """
  fp = av.DIR_PATH+"/data/KTAU_var_data/global_node_permutations_" + av.DATASET_NAME + "_tfrac_" + str(av.TEST_FRAC) + "_vfrac_" + str(av.VAL_FRAC) + ".pkl"
  if os.path.exists(fp) :
      with open(fp, 'rb') as f:
        all_info = pickle.load(f)
  else:
    all_info = {}
    for sample_frac in [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]:
      all_info[sample_frac] = {}
      for perm_type in ['rand','rev']:
        all_info[sample_frac][perm_type] = {}    
        no_of_nodes = gr.get_num_nodes()
        sample_size = int(sample_frac * no_of_nodes)      
        for n_perm in range(num_perms):
          all_info[sample_frac][perm_type][n_perm] = {}
          #global reordering of fraction of nodes
          node_map = np.arange(no_of_nodes)
          selected_nodes_for_perm = np.sort(np.random.choice(no_of_nodes, sample_size, replace=False))
          if perm_type == 'rand':
            node_map[selected_nodes_for_perm] = node_map[np.random.permutation(selected_nodes_for_perm)]
          else:
            rev_list = selected_nodes_for_perm[::-1].copy()  
            node_map[selected_nodes_for_perm] = node_map[rev_list]

          all_info[sample_frac][perm_type][n_perm]['node_map'] = node_map
          k_tau =  0
          for node in range(no_of_nodes):
            #fetch all neighbors of given node
            nbrs = sorted(list(gr.adjacency_list[node]))
            nbr_list_init = np.arange(len(nbrs))
            #new ids of nodes under global reordering
            nbrs_new = list(node_map[n] for n in nbrs)
            #permutation induced by the global reordering
            nbr_list = np.argsort(nbrs_new)
            all_info[sample_frac][perm_type][n_perm][node] = nbr_list
            if len(nbr_list_init)==1:
              k_tau=k_tau+0   
            else:    
              k_tau = k_tau +  stats.kendalltau(nbr_list, nbr_list_init)[0]
          all_info[sample_frac][perm_type][n_perm]['ktau'] = k_tau
    with open(fp, 'wb') as f:
      pickle.dump(all_info, f)
        
  return all_info

def compute_loss(av,gr, all_embeds):
    loss=0
    nodes = gr.query_node_list
    for i in range(len(nodes)):
      selfemb = all_embeds[nodes[i]]

      nbrs = all_embeds[list(gr.query_node_nbr[nodes[i]])]
      nonnbrs = all_embeds[list(gr.query_node_non_nbr[nodes[i]])]
      
      #https://pytorch.org/docs/master/generated/torch.nn.CosineSimilarity.html
      cos = nn.CosineSimilarity(dim=1, eps=1e-6)
      pos_scores = cos(nbrs,selfemb.unsqueeze(0))
      neg_scores = cos(nonnbrs,selfemb.unsqueeze(0))  
      
      len_pos = pos_scores.shape[0]
      len_neg = neg_scores.shape[0]
      expanded_pos_scores = pos_scores.unsqueeze(1).expand(len_pos,len_neg)
      expanded_neg_scores = neg_scores.unsqueeze(0).expand(len_pos,len_neg)
        
      loss += torch.max(av.MARGIN + expanded_neg_scores - expanded_pos_scores,cudavar(av,torch.tensor([0.]))).sum()

    return loss.item()

def performance_analysis(av,gr: PermGnnGraph):
  query_nodes, \
    list_training_edges, \
    list_training_non_edges, \
    list_test_edges, \
    list_test_non_edges, \
    list_val_edges, \
    list_val_non_edges = fetch_lp_data_split(av,gr)

  prep_permgnn_graph(av,gr,query_nodes,list_training_edges,list_training_non_edges,list_val_edges,list_test_edges,list_val_non_edges,list_test_non_edges)
  
  #num_perms = 5
  num_perms = 1
  all_info = generate_global_permutations(av,gr,num_perms) 

  device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
  permNet = PermutationGenerator(av,gr).to(device)
  permGNN = PermutationInvariantGNN(av,gr,permNet).to(device)
  #if VAL_FRAC is 0, we fetch model weights from last trained epoch
  # else we fetch  best performing model on validation dataset
  if av.VAL_FRAC==0:
    checkpoint = load_model(av)
    logger.info("Loading latest trained model from training epoch %d",checkpoint['epoch'])
  else:
    es = EarlyStoppingModule(av)
    checkpoint = es.load_best_model()
    logger.info("Loading best validation result model from training epoch %d",checkpoint['epoch'])

  permGNN.load_state_dict(checkpoint['model_state_dict'])

  cos = nn.CosineSimilarity(dim=1, eps=1e-6)
  
  all_nodes = list(range(permGNN.gr.get_num_nodes()))

  canonical_lstm_op = cudavar(av,torch.tensor([]))
  canonical_embeds = cudavar(av,torch.tensor([]))
  #batch and send nodes to avoid memory limit crash for larger graphs
  for i in range(0,permGNN.gr.get_num_nodes(),av.BATCH_SIZE) : 
    batch_nodes = all_nodes[i:i+av.BATCH_SIZE]
    set_size = permGNN.permNet.set_size_all[batch_nodes]
    neighbour_features = permGNN.permNet.padded_neighbour_features_all[batch_nodes]
    lstm_op,embeds = permGNN.getEmbeddingForFeatures(set_size,neighbour_features,True)
    canonical_lstm_op = torch.cat((canonical_lstm_op,lstm_op),dim=0)
    canonical_embeds = torch.cat((canonical_embeds,embeds),dim=0)
  canonical_inputs = permGNN.permNet.padded_neighbour_features_all.flatten(1) 
   
  canonical_tr_loss = compute_loss(av,gr,canonical_embeds)   
  
  for sample_frac in [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]:
    for perm_type in ['rand','rev']:
      for n_perm in range(num_perms):
        perm_info = all_info[sample_frac][perm_type][n_perm]  
        all_embeds = cudavar(av,torch.tensor([]))
        all_lstm_op = cudavar(av,torch.tensor([]))
        #permute neighbour features
        perm_neighbour_features = []
        for node in range(gr.get_num_nodes()): 
          node_feats_orig = permGNN.permNet.padded_neighbour_features_all[node]
          node_feats_perm = node_feats_orig[torch.tensor(perm_info[node])]
          perm_neighbour_features.append(node_feats_perm)
        perm_neighbour_features = pad_sequence(perm_neighbour_features,batch_first=True)
        #batch and send nodes to avoid memory limit crash for larger graphs
        for i in range(0,permGNN.gr.get_num_nodes(),av.BATCH_SIZE) : 
          batch_nodes = all_nodes[i:i+av.BATCH_SIZE]
          set_size = permGNN.permNet.set_size_all[batch_nodes]
          neighbour_features = perm_neighbour_features[batch_nodes]
          lstm_op,embeds = permGNN.getEmbeddingForFeatures(set_size,neighbour_features,True)
          all_lstm_op = torch.cat((all_lstm_op,lstm_op),dim=0)
          all_embeds = torch.cat((all_embeds,embeds),dim=0)
        
        all_info[sample_frac][perm_type][n_perm]['inputs_sens_score_list'] = cos(canonical_inputs, perm_neighbour_features.flatten(1))
        all_info[sample_frac][perm_type][n_perm]['lstm_op_sens_score_list'] = cos(canonical_lstm_op,all_lstm_op) 
        all_info[sample_frac][perm_type][n_perm]['embeds_sens_score_list'] = cos(canonical_embeds,all_embeds)

        perm_tr_loss = compute_loss(av,gr,all_embeds)
        all_info[sample_frac][perm_type][n_perm]['loss_var'] = abs(perm_tr_loss-canonical_tr_loss)/canonical_tr_loss
  fname = av.DIR_PATH+"/data/KTAU_var_data/" + "Ktau_variation_data"+"_"+av.TASK+"_"+av.DATASET_NAME+"_tfrac_"+str(av.TEST_FRAC)+"_vfrac_"+str(av.VAL_FRAC) + "_data.pkl"
  pickle.dump(all_info,open(fname,"wb"))

def  plot_permgnn_permutation_sensitivity_across_layers(av,gr: PermGnnGraph):
  ktau_list = {}
  inputs_sens_list = {}
  lstm_op_sens_list = {}
  embeds_sens_list = {}  
  #num_perms=5
  num_perms=1
  for task in ['PermGNN']:
    fp = av.DIR_PATH+"/data/KTAU_var_data/" + "Ktau_variation_data"+"_"+task+"_"+av.DATASET_NAME+"_tfrac_"+str(av.TEST_FRAC)+"_vfrac_"+str(av.VAL_FRAC) + "_data.pkl"
    info = pickle.load(open(fp,"rb"))
    ktau_list[task] = []
    inputs_sens_list[task] = []
    lstm_op_sens_list[task] = []
    embeds_sens_list[task] = []      
    for sample_frac in [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]:
      for perm_type in ['rand','rev']:
        for n_perm in range(num_perms):
          ktau_list[task].append(info[sample_frac][perm_type][n_perm]['ktau'])
          inputs_sens_list[task].append(torch.mean(info[sample_frac][perm_type][n_perm]['inputs_sens_score_list']).tolist())
          lstm_op_sens_list[task].append(torch.mean(info[sample_frac][perm_type][n_perm]['lstm_op_sens_score_list']).tolist())
          embeds_sens_list[task].append(torch.mean(info[sample_frac][perm_type][n_perm]['embeds_sens_score_list']).tolist())

  for t in ['PermGNN'] :   
    ktau_list[t] = [float(i) for i in ktau_list[t]]  
    ktau_list[t] = [i/max(ktau_list[t]) for i in ktau_list[t]] 
  plt.clf() 
  plt.scatter(ktau_list['PermGNN'], inputs_sens_list['PermGNN'], c="g")
  plt.scatter(ktau_list['PermGNN'], lstm_op_sens_list['PermGNN'], c="r")
  plt.scatter(ktau_list['PermGNN'], embeds_sens_list['PermGNN'], c="b")
  plt.ylabel('Permutation Insensitivity')
  plt.xlabel('KTAU')  
  plt.title(av.DATASET_NAME + "-PermutationInsensitivityKtau" )
  fp = av.DIR_PATH + "/data/KTAU_var_data/" + av.DATASET_NAME + "-PermutationInsensitivityKtau"
  plt.savefig(fp, bbox_inches='tight')

def plot_trLoss_vs_ktau(av,gr: PermGnnGraph):
  trLoss_list = {}
  ktau_list = {}
  #num_perms=5
  num_perms=1
  for task in ['PermGNN', 'Multiperm','1Perm']:
    fp = av.DIR_PATH+"/data/KTAU_var_data/" + "Ktau_variation_data"+"_"+task+"_"+av.DATASET_NAME+"_tfrac_"+str(av.TEST_FRAC)+"_vfrac_"+str(av.VAL_FRAC) + "_data.pkl"
    info = pickle.load(open(fp,"rb"))
    trLoss_list[task] = [] 
    ktau_list[task] = []
    for sample_frac in [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]:
      for perm_type in ['rand','rev']:
        for n_perm in range(num_perms):
          trLoss_list[task].append(info[sample_frac][perm_type][n_perm]['loss_var'])
          ktau_list[task].append(info[sample_frac][perm_type][n_perm]['ktau'])
  for t in ['PermGNN', 'Multiperm','1Perm'] :   
    ktau_list[t] = [float(i) for i in ktau_list[t]]  
    ktau_list[t] = [i/max(ktau_list[t]) for i in ktau_list[t]]  
  
  plt.clf() 
  plt.scatter(ktau_list['PermGNN'], trLoss_list['PermGNN'], c="g")
  plt.scatter(ktau_list['1Perm'], trLoss_list['1Perm'], c="r")
  plt.scatter(ktau_list['Multiperm'], trLoss_list['Multiperm'], c="b")
  plt.ylabel('MAP')
  plt.xlabel('KTAU')  
  plt.title(av.DATASET_NAME + "-MAPvsKtau" )
  fp = av.DIR_PATH + "/data/KTAU_var_data/" + av.DATASET_NAME + "-MAPvsKtau"
  plt.savefig(fp, bbox_inches='tight')
 
if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--logpath",                 type=str,   default="data/logDir/logfile",help="/path/to/log")
  ap.add_argument("--want_cuda",               type=bool,  default=True)
  ap.add_argument("--has_cuda",                type=bool,  default=torch.cuda.is_available())
  ap.add_argument("--SINKHORN_TEMP",           type=float, default=0.5)
  ap.add_argument("--SINKHORN_ITER",           type=int,   default=10)
  ap.add_argument("--PERM_NETWORK_LATENT_DIM", type=int,   default=16)
  ap.add_argument("--NOISE_FACTOR",            type=float, default=1)
  ap.add_argument("--NUM_PERMS",               type=int,   default=10)
  ap.add_argument("--LSTM_HIDDEN_DIM",         type=int,   default=32)
  ap.add_argument("--EMBEDDING_DIM",           type=int,   default=16)
  ap.add_argument("--NUM_EPOCHS",              type=int,   default=2)
  ap.add_argument("--BATCH_SIZE",              type=int,   default=128)
  ap.add_argument("--MARGIN",                  type=float, default=.01)
  ap.add_argument("--LEARNING_RATE_FUNC",      type=float, default=0.00005)
  ap.add_argument("--LEARNING_RATE_PERM",      type=float, default=0.00005)
  ap.add_argument("--DIR_PATH",                type=str,   default=".",help="path/to/datasets")
  ap.add_argument("--DATASET_NAME",            type=str,   default="Twitter_3", help="cora/citeseer/Twitter_3/Gplus_1/PB")
  ap.add_argument("--OPTIM",                   type=str,   default="SGD", help="SGD/Adam")
  ap.add_argument("--SCORE",                   type=str,   default="AUC_AP", help="AUC_AP/MAP")
  ap.add_argument("--RESUME_RUN",              type=bool,  default=False, help="If set to True, be sure to mention TASK")
  ap.add_argument("--TEST_FRAC",               type=float, default=0.4)
  ap.add_argument("--VAL_FRAC",                type=float, default=0.1)
  ap.add_argument("--TOP_K",                   type=int,   default=0)
  ap.add_argument("--TASK",                    type=str,   default="LP",help="1Perm/Multiperm/PermGNN")

  av = ap.parse_args()
  av.logpath = av.logpath+"_performance_variation_wrt_ktau"+"_"+av.TASK+"_"+av.DATASET_NAME+"_tfrac_"+str(av.TEST_FRAC)+"_vfrac_"+str(av.VAL_FRAC)
  set_log(av)
  logger.info("Command line")  
  logger.info('\n'.join(sys.argv[:]))  
  if av.DATASET_NAME in ['Twitter_3','Gplus_1','PB']:
    gr = TwitterGraph(av)
  else:
    gr = LinqsUmdDocumentGraph(av)
  for task  in ['PermGNN', 'Multiperm','1Perm']:
    av.TASK = task  
    pickle_fp =  av.DIR_PATH + "/data/KTAU_var_data/" + "Ktau_variation_data"+"_"+av.TASK+"_"+av.DATASET_NAME+"_tfrac_"+str(av.TEST_FRAC)+"_vfrac_"+str(av.VAL_FRAC) + "_data.pkl"
    if not os.path.exists(pickle_fp):
      performance_analysis(av,gr)
  
  plot_trLoss_vs_ktau(av,gr)
  plot_permgnn_permutation_sensitivity_across_layers(av,gr)

      
  #python -m minmax.performance_variation_wrt_ktau --TEST_FRAC=0.4 --VAL_FRAC=0.1 -DATASET_NAME="cora"
