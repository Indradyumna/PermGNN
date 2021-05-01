import argparse
import numpy as np
import random
import time
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from common import logger, set_log
from minmax.graphs import PermGnnGraph, LinqsUmdDocumentGraph 
from minmax.utils import cudavar,load_model,save_model,init_optimizers,set_learnable_parameters,pytorch_sample_gumbel

class PermutationGenerator(nn.Module):
  """
    Returns a scrambled(permutated) sequence of neighbours features for each node
  """
  def __init__(self, av, gr: PermGnnGraph):
    super(PermutationGenerator, self).__init__()
    self.av = av
    self.gr = gr
    self.features = nn.Embedding(self.gr.get_num_nodes() ,self.gr.get_num_features())
    self.features.weight = nn.Parameter(cudavar(self.av,torch.FloatTensor(self.gr.node_features)),requires_grad=False)
    self.feature_dim = self.gr.get_num_features()
        
    self.adj_list = self.gr.adjacency_list
    #Max set size is one greater than max node outdegree accounting for presence of the node itself in set
    self.max_set_size = self.gr.get_max_node_outdegree() + 1 
    # Lookup table set_size:mask . Given set_size k and max_set_size n
    # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
    self.set_size_to_mask_map = [torch.cat((torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([x,self.max_set_size-x])).repeat(x,1),
                                 torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([0,self.max_set_size])).repeat(self.max_set_size-x,1)))
                                 for x in range(1,self.max_set_size+1)]
    # List of tensors corr to each node. Each tensor is input sequence of neighbourhood features
    #if self.av.TASK == "1Perm":
    self.neighbour_features_all = [self.features(cudavar(self.av,torch.LongTensor(sorted(list(self.adj_list[node]))))) for node in range(self.gr.get_num_nodes())]
    #else:
    #  self.neighbour_features_all = [self.features(cudavar(self.av,torch.LongTensor(list(self.adj_list[node])))) for node in range(self.gr.get_num_nodes())]
    #numpy array of set sizes for all node ids. Used later for variable length LSTM code
    self.set_size_all = np.array([len(x) for x in self.neighbour_features_all])
    #Generate boolean mask for each node based on it's set_size. Used for masked sinkhorn normalization 
    self.sets_maskB_all = cudavar(self.av,torch.stack([self.set_size_to_mask_map[x-1]==0 for x in self.set_size_all]))
    #Generates padded tensor of dim(num_nodes*max_set_size*feature_dimension)
    self.padded_neighbour_features_all = pad_sequence(self.neighbour_features_all,batch_first=True)
    self.latent_dim = self.av.PERM_NETWORK_LATENT_DIM
    self.output_dim = self.max_set_size
    self.linear1 = nn.Linear(self.feature_dim, self.latent_dim)
    self.relu1 = nn.ReLU()
    self.linear2 = nn.Linear(self.latent_dim, self.output_dim)

  def pytorch_sinkhorn_iters_mask(self, log_alpha,maskB,temp,noise_factor=1.0, n_iters = 20):
    batch_size = log_alpha.size()[0]
    n = log_alpha.size()[1]
    log_alpha = log_alpha.view(-1, n, n)
    noise = pytorch_sample_gumbel(self.av,[batch_size, n, n])*noise_factor
    log_alpha = log_alpha + noise
    log_alpha = torch.div(log_alpha,temp)

    for i in range(n_iters):
      log_alpha = (log_alpha - (torch.logsumexp(log_alpha.masked_fill_(maskB,float('-inf')), dim=2, keepdim=True)).view(-1, n, 1)).masked_fill_(maskB,float('-inf'))

      log_alpha = (log_alpha - (torch.logsumexp(log_alpha.masked_fill_(maskB,float('-inf')), dim=1, keepdim=True)).view(-1, 1, n)).masked_fill_(maskB,float('-inf'))
    return torch.exp(log_alpha)

  def forward(self, nodes):
    """
      :param   nodes    : batch of node ids from range 0..NUM_NODES
      :return  set_size : neighbourhood set sizes for each node. Needed for variable lenghth LSTM code
      :return  permuted_neighbour_features : permutation of neighbour feature vectors for each node
                                             For node_set_size k and max_set_size n last (n-k) rows are padded with 0
    """
    set_size = self.set_size_all[nodes]
    padded_neighbour_features = self.padded_neighbour_features_all[nodes]
    neighbour_features_flat = padded_neighbour_features.view(-1, self.feature_dim)
    net = self.linear2(self.relu1(self.linear1(neighbour_features_flat)))
    pre_sinkhorn = net.view(-1,self.output_dim, self.output_dim)
    post_sinkhorn = self.pytorch_sinkhorn_iters_mask(pre_sinkhorn,self.sets_maskB_all[nodes],temp=self.av.SINKHORN_TEMP,noise_factor=self.av.NOISE_FACTOR,n_iters=self.av.SINKHORN_ITER)
    permuted_neighbour_features = torch.matmul(post_sinkhorn.permute(0,2,1),padded_neighbour_features)
    return set_size,permuted_neighbour_features
 
class PermutationInvariantGNN(nn.Module):
  """
    PermutationGenerator provides permuted neighbourhood features for each node
    Variable length LSTM+FC layer generates embedding for each node
    Computes Hinge ranking loss
  """
  def __init__(self, av,gr: PermGnnGraph, permNet: PermutationGenerator):
    super(PermutationInvariantGNN, self).__init__()
    self.av = av
    self.gr = gr
    self.features = nn.Embedding(self.gr.get_num_nodes() ,self.gr.get_num_features())
    self.features.weight = nn.Parameter(cudavar(self.av,torch.FloatTensor(self.gr.node_features)),requires_grad=False) 
    self.adj_list = self.gr.adjacency_list

    self.lstm_input_size  = self.gr.get_num_features()
    self.lstm_hidden_size = self.av.LSTM_HIDDEN_DIM
    self.fclayer_output_size = self.av.EMBEDDING_DIM
    #LSTM layer init
    self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, num_layers=1,batch_first=True)
    #FC layer init. Bias folded in with Weight matrix, so aug_lstm_output generated below to compensate 
    self.fully_connected_layer = nn.Linear(self.lstm_hidden_size+1, self.fclayer_output_size,bias=False)
    self.permNet = permNet
 
  def forward(self, nodes): 
    """
      av.TASK  1Perm        : for LP . bypass permNet. Use canonical input sequence
      av.TASK  Multiperm    : for LP . bypass permNet. permute inpute sequence every time 
                                       randomly using torch.randperm 
      av.TASK  PermGNN      : use permNet to create scrambling of input sequence                                 
    """
    if self.av.TASK =="1Perm" :
      set_size = self.permNet.set_size_all[nodes]
      neighbour_features = self.permNet.padded_neighbour_features_all[nodes]
    elif self.av.TASK=="Multiperm":
      set_size = self.permNet.set_size_all[nodes]
      neighbour_features = pad_sequence([mat[torch.randperm(int(size))] \
                                            for (mat,size) in zip(self.permNet.padded_neighbour_features_all[nodes],\
                                                                  self.permNet.set_size_all[nodes])],\
                                           batch_first=True)
    else:   
      set_size,neighbour_features = self.permNet(nodes)

    return self.getEmbeddingForFeatures(set_size,neighbour_features)

  def getEmbeddingForFeatures(self, set_size,neighbour_features,diagnostic_mode=False):
    """
      :param  set_size           : neighbourhood set sizes for each node. 
                                   Needed for variable lenghth LSTM code
      :param  neighbour_features : permutation of neighbour feature vectors for each node
                                   For node_set_size k and max_set_size n last (n-k) rows are padded with 0
      :return node_embeddings    : Embedding dim currently same as input feature dimension. 
    """
    #Below 3 steps of pack_padded_sequence -> LSTM -> pad_packed_sequence 
    #ensures variable length LSTM. So 0 padded rows not fed to LSTM network
    packed_neighbour_features = pack_padded_sequence(neighbour_features,set_size,batch_first=True,enforce_sorted=False)
    packed_lstm_output, (ht, ct) = self.lstm(packed_neighbour_features)
    padded_lstm_output = pad_packed_sequence(packed_lstm_output,batch_first=True)
    #appends 1 in bias column except for the rows which are pads
    aug_lstm_output = torch.cat((padded_lstm_output[0],pad_sequence([cudavar(self.av,torch.ones([x])).unsqueeze(0).t() for x in padded_lstm_output[1].tolist()], batch_first=True)),dim=2)
    node_embeddings = torch.sum(self.fully_connected_layer(aug_lstm_output),dim=1) 
    #diagonistic mode wasadded later to instrument sensitivity to permutations across layers
    if diagnostic_mode:
      lstm_output_flat = padded_lstm_output[0].flatten(1)
      zero_pad =  cudavar(self.av,torch.zeros(padded_lstm_output[0].shape[0],(self.permNet.max_set_size-padded_lstm_output[0].shape[1])*padded_lstm_output[0].shape[2]))
      final = torch.cat((lstm_output_flat,zero_pad),1)
      return final.data,node_embeddings.data
    else:
      return node_embeddings

  def computeLoss(self, nodes):
    """
      :param   nodes  : batch of node ids from range 0..NUM_NODES
      :return  loss   : Hinge ranking loss
    """
    loss = 0
    all_nodes = list(range(self.gr.get_num_nodes()))
    all_embeds = cudavar(self.av,torch.tensor([]))

    if self.av.TASK=="Multiperm":
      all_embeds_perms = []
      for rep in range(self.av.NUM_PERMS):
        temp = cudavar(self.av,torch.tensor([]))
        #batch and send nodes to avoid memory limit crash for larger graphs
        for i in range(0,self.gr.get_num_nodes(),self.av.BATCH_SIZE) :
          batch_nodes = all_nodes[i:i+self.av.BATCH_SIZE]
          temp = torch.cat((temp,self.forward(batch_nodes)),dim=0)
        all_embeds_perms.append(temp)
      all_embeds = torch.mean(torch.stack(all_embeds_perms),0)
    else:
      #batch and send nodes to avoid memory limit crash for larger graphs
      for i in range(0,self.gr.get_num_nodes(),self.av.BATCH_SIZE) :
        batch_nodes = all_nodes[i:i+self.av.BATCH_SIZE]
        all_embeds = torch.cat((all_embeds,self.forward(batch_nodes)),dim=0)

    #Filter for query_nodes
    nodes = list(set(self.gr.query_node_list).intersection(set(nodes)))
    
    for i in range(len(nodes)):
      selfemb = all_embeds[nodes[i]]

      nbrs = all_embeds[list(self.gr.query_node_nbr[nodes[i]])]
      nonnbrs = all_embeds[list(self.gr.query_node_non_nbr[nodes[i]])]
      
      #https://pytorch.org/docs/master/generated/torch.nn.CosineSimilarity.html
      cos = nn.CosineSimilarity(dim=1, eps=1e-6)
      pos_scores = cos(nbrs,selfemb.unsqueeze(0))
      neg_scores = cos(nonnbrs,selfemb.unsqueeze(0))  
      
      len_pos = pos_scores.shape[0]
      len_neg = neg_scores.shape[0]
      expanded_pos_scores = pos_scores.unsqueeze(1).expand(len_pos,len_neg)
      expanded_neg_scores = neg_scores.unsqueeze(0).expand(len_pos,len_neg)
        
      loss += torch.max(self.av.MARGIN + expanded_neg_scores - expanded_pos_scores,cudavar(self.av,torch.tensor([0.]))).sum()

    return loss

