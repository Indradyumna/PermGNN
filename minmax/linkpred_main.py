import argparse
import numpy as np
import random
import time
import sys
import os
import pickle

import torch
import torch.nn as nn
from torch.nn.utils.rnn import  pad_sequence

from common import logger, set_log
from minmax.graphs import PermGnnGraph, LinqsUmdDocumentGraph, TwitterGraph,ABGraph
from minmax.main import PermutationGenerator, PermutationInvariantGNN
from minmax.linkpred_utils import prep_permgnn_graph, get_lp_scores, compute_scores_from_embeds, fetch_lp_data_split
from minmax.utils import cudavar,load_model,save_model,init_optimizers,set_learnable_parameters
from minmax.earlystopping import EarlyStoppingModule

def fetch_permgnn_embeddings(av,gr: PermGnnGraph):
  avTask = av.TASK

  av.TASK = "PermGNN"
  pickle_fp = "./data/embeddingPickles/"+av.TASK+"_"+av.DATASET_NAME+"_tfrac_"+str(av.TEST_FRAC)+"_vfrac_"+str(av.VAL_FRAC) + "_embedding_mat.pkl"
  if not os.path.exists(pickle_fp):
    query_nodes, \
    list_training_edges, \
    list_training_non_edges, \
    list_test_edges, \
    list_test_non_edges, \
    list_val_edges, \
    list_val_non_edges = fetch_lp_data_split(av,gr)

    prep_permgnn_graph(av,gr,query_nodes,list_training_edges,list_training_non_edges,list_val_edges,list_test_edges,list_val_non_edges,list_test_non_edges)
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

    all_nodes = list(range(permGNN.gr.get_num_nodes()))
    all_embeds = cudavar(av,torch.tensor([]))
    for i in range(0,permGNN.gr.get_num_nodes(),av.BATCH_SIZE) :
      batch_nodes = all_nodes[i:i+av.BATCH_SIZE]
      set_size = permGNN.permNet.set_size_all[batch_nodes]
      neighbour_features = permGNN.permNet.padded_neighbour_features_all[batch_nodes]
      all_embeds = torch.cat((all_embeds,permGNN.getEmbeddingForFeatures(set_size,neighbour_features).data),dim=0)

    logger.info("Creating permgnn embedding pickle at %s",pickle_fp)
    with open(pickle_fp, 'wb') as f:
      pickle.dump(all_embeds, f)

  else:
    logger.info("Loading permgnn embedding pickle from %s",pickle_fp)
    with open(pickle_fp, 'rb') as f:
      all_embeds = pickle.load(f)

  av.TASK = avTask
  return cudavar(av,all_embeds)


def log_scores(av,permGNN,query_nodes,list_val_edges, list_val_non_edges, list_test_edges,list_test_non_edges,start_time,epochLoss,epoch,phase):
  score_list = []
  if av.SCORE == "MAP":
    if av.VAL_FRAC == 0:
      map_score = get_lp_scores(av,permGNN,query_nodes,list_test_edges,list_test_non_edges)
    else:
      map_score = get_lp_scores(av,permGNN,query_nodes,list_val_edges,list_val_non_edges)
    end_time = time.time()
    logger.info("Epoch: %d %s phase loss: %f map_score: %.6f Time: %.2f",epoch,phase,epochLoss,map_score, end_time-start_time)
    score_list=[map_score]
  elif av.SCORE == "AUC_AP":
    if av.VAL_FRAC == 0:
      auc_score,ap_score = get_lp_scores(av,permGNN,query_nodes,list_test_edges,list_test_non_edges)
    else:
      auc_score,ap_score = get_lp_scores(av,permGNN,query_nodes,list_val_edges,list_val_non_edges)
    end_time = time.time()
    logger.info("Epoch: %d %s phase loss: %f auc_score: %.6f ap_score: %.6f Time: %.2f",epoch,phase,epochLoss,auc_score,ap_score, end_time-start_time)
    score_list = [auc_score,ap_score]
  else:
    raise NotImplementedError()
  return score_list 
 

def run_graph_lp(av,gr: PermGnnGraph):
  #if av.has_cuda:
  #  torch.cuda.reset_max_memory_allocated(0)
  query_nodes, \
    list_training_edges, \
    list_training_non_edges, \
    list_test_edges, \
    list_test_non_edges, \
    list_val_edges, \
    list_val_non_edges = fetch_lp_data_split(av,gr)

  prep_permgnn_graph(av,gr,query_nodes,list_training_edges,list_training_non_edges,list_val_edges,list_test_edges,list_val_non_edges,list_test_non_edges)

  ###permGNN part starts
  device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
  permNet = PermutationGenerator(av,gr).to(device)
  permGNN = PermutationInvariantGNN(av,gr,permNet).to(device)
  
  es = EarlyStoppingModule(av)

  optimizerPerm,optimizerFunc = init_optimizers(av,permGNN)
  starting_epoch = 0 
  
  #if True, load latest epoch model and optimizer state and resume training 
  #if False, train from scratch
  if (av.RESUME_RUN):
    checkpoint = load_model(av)
    permGNN.load_state_dict(checkpoint['model_state_dict'])
    optimizerPerm.load_state_dict(checkpoint['optimizer_perm_state_dict'])
    optimizerFunc.load_state_dict(checkpoint['optimizer_func_state_dict'])
    starting_epoch = checkpoint['epoch'] + 1
    #NOTE:av.RESUME_RUN will becom False, but currently it's unused anywhere else
    av = checkpoint['av']

  nodes = list(range(gr.get_num_nodes()))
  #for epoch in range(starting_epoch,av.NUM_EPOCHS):
  epoch = starting_epoch
  #if VAL_FRAC is 0, we train model for NUM_EPOCHS
  #else we train model till early stopping criteria is met
  while av.VAL_FRAC!=0 or epoch<av.NUM_EPOCHS:
    random.shuffle(nodes)
    if av.TASK != "1Perm" and av.TASK != "Multiperm":
      start_time = time.time()
      set_learnable_parameters(permGNN,isMaxPhase=True)

      epochLoss=0
      for i in range(0, gr.get_num_nodes(), av.BATCH_SIZE):
        nodes_batch = nodes[i:i+av.BATCH_SIZE]
        permGNN.zero_grad()
        loss = -permGNN.computeLoss(nodes_batch)
        if loss==0:
            continue
        loss.backward()
        optimizerPerm.step()
        epochLoss = epochLoss + loss.item()
      score_list = log_scores(av,permGNN,query_nodes,list_val_edges, list_val_non_edges,list_test_edges,list_test_non_edges,start_time,epochLoss,epoch,phase="max")

    start_time = time.time()
    set_learnable_parameters(permGNN,isMaxPhase=False)

    epochLoss = 0 
    for i in range(0, gr.get_num_nodes(), av.BATCH_SIZE):
      nodes_batch = nodes[i:i+av.BATCH_SIZE]
      permGNN.zero_grad()
      loss = permGNN.computeLoss(nodes_batch)
      if loss==0:
          continue
      loss.backward()
      optimizerFunc.step()
      epochLoss = epochLoss + loss.item()       
    score_list = log_scores(av,permGNN,query_nodes,list_val_edges, list_val_non_edges,list_test_edges,list_test_non_edges,start_time,epochLoss,epoch,phase="min")

    save_model(av,permGNN,optimizerPerm, optimizerFunc, epoch, saveAllEpochs=False)
    if av.VAL_FRAC!=0:
      if es.check(score_list,permGNN,epoch):
        break
    epoch+=1
  if av.has_cuda:
    logger.info("Max gpu memory used: %.6f ",torch.cuda.max_memory_allocated(device=0)/(1024**3))

def lp_permute_test_result(av,gr: PermGnnGraph):
  query_nodes, \
    list_training_edges, \
    list_training_non_edges, \
    list_test_edges, \
    list_test_non_edges, \
    list_val_edges, \
    list_val_non_edges = fetch_lp_data_split(av,gr)

  prep_permgnn_graph(av,gr,query_nodes,list_training_edges,list_training_non_edges,list_val_edges,list_test_edges,list_val_non_edges,list_test_non_edges)

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

  logger.info("Test scores  with canonical input sequence")
  start_time = time.time()
  
  all_nodes = list(range(permGNN.gr.get_num_nodes()))
  all_embeds = cudavar(av,torch.tensor([]))
  #batch and send nodes to avoid memory limit crash for larger graphs
  for i in range(0,permGNN.gr.get_num_nodes(),av.BATCH_SIZE) : 
    batch_nodes = all_nodes[i:i+av.BATCH_SIZE]
    set_size = permGNN.permNet.set_size_all[batch_nodes]
    neighbour_features = permGNN.permNet.padded_neighbour_features_all[batch_nodes]
    all_embeds = torch.cat((all_embeds,permGNN.getEmbeddingForFeatures(set_size,neighbour_features).data),dim=0)

  auc_score, ap_score, map_score, mrr_score = compute_scores_from_embeds(av,all_embeds,query_nodes,list_test_edges,list_test_non_edges)

  end_time = time.time()
  logger.info("auc_score: %.6f ap_score: %.6f map_score: %.6f mrr_score: %.6f Time: %.2f",auc_score,ap_score,map_score,mrr_score ,end_time-start_time)

  logger.info("Test scores with randomly permuted input sequence")
  for num_run in range(10):
    start_time = time.time()
  
    all_nodes = list(range(permGNN.gr.get_num_nodes()))
    all_embeds = cudavar(av,torch.tensor([]))
    #permute neighbour features
    perm_neighbour_features = pad_sequence([mat[torch.randperm(int(size))] \
                                            for (mat,size) in zip(permGNN.permNet.padded_neighbour_features_all,permGNN.permNet.set_size_all)],\
                                           batch_first=True)
    #batch and send nodes to avoid memory limit crash for larger graphs
    for i in range(0,permGNN.gr.get_num_nodes(),av.BATCH_SIZE) : 
      batch_nodes = all_nodes[i:i+av.BATCH_SIZE]
      set_size = permGNN.permNet.set_size_all[batch_nodes]
      #neighbour_features = permGNN.padded_neighbour_features_all[batch_nodes]
      neighbour_features = perm_neighbour_features[batch_nodes]      
      all_embeds = torch.cat((all_embeds,permGNN.getEmbeddingForFeatures(set_size,neighbour_features).data),dim=0)

    auc_score, ap_score, map_score, mrr_score = compute_scores_from_embeds(av,all_embeds,query_nodes,list_test_edges,list_test_non_edges)

    end_time = time.time()
    logger.info("auc_score: %.6f ap_score: %.6f map_score: %.6f mrr_score: %.6f Time: %.2f",auc_score,ap_score,map_score,mrr_score ,end_time-start_time)

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
  ap.add_argument("--ONLY_PREDICT",            type=bool,  default=False)
  ap.add_argument("--TASK",                    type=str,   default="LP",help="1Perm/Multiperm/PermGNN")

  av = ap.parse_args()
  av.logpath = av.logpath+"_"+av.TASK+"_"+av.DATASET_NAME+"_tfrac_"+str(av.TEST_FRAC)+"_vfrac_"+str(av.VAL_FRAC)
  set_log(av)
  logger.info("Command line")  
  logger.info('\n'.join(sys.argv[:]))  
  if av.DATASET_NAME in ['AB100','AB1k','AB5k','AB10k','AB20k','AB50k']:
    av.VAL_FRAC=0.0
    #av.NUM_EPOCHS=1
    #av.SINKHORN_ITER=1
    #av.NOISE_FACTOR=0
    gr = ABGraph(av)
  elif av.DATASET_NAME in ['Twitter_3','Gplus_1','PB']:
    gr = TwitterGraph(av)
  else:
    gr = LinqsUmdDocumentGraph(av)
  if not av.ONLY_PREDICT:
    run_graph_lp(av,gr)
  logger.info("Running permute test lp")
  lp_permute_test_result(av,gr)
