import numpy as np
import networkx as nx
from collections import defaultdict
import random
import time
import os

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, ndcg_score

from common import logger, set_log
from minmax.graphs import PermGnnGraph
from minmax.utils import cudavar

import heapq
from itertools import combinations 


class LSH(object):
  """
    Fill
  """
  def __init__(self, av,gr: PermGnnGraph, num_hash_tables = 10, subset_size=8 ):
    super(LSH, self).__init__()
    assert(subset_size<=av.HASHCODE_DIM)
    self.av = av
    self.gr = gr
    self.all_nodes = self.gr.get_num_nodes()
  
    self.num_hash_tables = num_hash_tables
    #No. of buckets in a hashTable is 2^subset_size
    self.subset_size = subset_size
    self.powers_of_two = cudavar(self.av,torch.from_numpy(1 << np.arange(self.subset_size - 1, -1, -1)).type(torch.FloatTensor))
    self.hash_functions = None
    self.init_hash_functions()

    #This contains +1 or -1. used for bucketifying
    self.hashcode_mat = cudavar(self.av,torch.tensor([])) 
    self.all_hash_tables = []
    self.candidate_set = np.zeros((self.gr.get_num_nodes(),self.gr.get_num_nodes()))

  def init_hash_functions(self):
    self.hash_functions = cudavar(self.av,torch.LongTensor([]))

    hash_code_dim = self.av.HASHCODE_DIM
    indices = list(range(hash_code_dim))
    for i in range(self.num_hash_tables):
      random.shuffle(indices)
      self.hash_functions= torch.cat((self.hash_functions,cudavar(self.av,torch.LongTensor([indices[:self.subset_size]]))),dim=0)

  def init_hash_code_mat(self,all_hashcodes): 
    self.hashcode_mat = cudavar(self.av,torch.sign(all_hashcodes))
    if (torch.sign(all_hashcodes)==0).any(): 
      logger.info("Hashcode had 0 bits. replacing all with 1")
      all_hashcodes[all_hashcodes==0]=1

  def init_candidate_set(self,list_edges, list_non_edges): 
    for (a,b) in list_non_edges: 
      self.candidate_set[a][b] = -1
      self.candidate_set[b][a] = -1 
    for (a,b) in list_edges: 
      self.candidate_set[a][b] = 1
      self.candidate_set[b][a] = 1
       


  def assign_bucket(self,function_id,node_hash_code):
    func = self.hash_functions[function_id]
    # convert sequence of -1 and 1 to binary by replacing -1 s to 0
    binary_id = torch.max(torch.index_select(node_hash_code,dim=0,index=func),cudavar(self.av,torch.tensor([0.])))
    #map binary sequence to int which is bucket Id
    bucket_id = self.powers_of_two@binary_id
    return bucket_id.item()

  def bucketify(self): 
    """
      For all hash funcitons: 
        Loop over all nodes
        Assign node to bucket in hash table corr. to hash function 
    """ 
    
    self.all_hash_tables = []
    for func_id in range(self.num_hash_tables): 
      hash_table = {}
      for id in range(2**self.subset_size): 
        hash_table[id] = []
      for node in range(self.gr.get_num_nodes()):
        hash_table[self.assign_bucket(func_id,self.hashcode_mat[node])].append(node)
      self.all_hash_tables.append(hash_table)
        
  def pretty_print_hash_tables(self,topk): 
    for table_id in range(self.num_hash_tables): 
      len_list = sorted([len(self.all_hash_tables[table_id][bucket_id]) for bucket_id in range(2**self.subset_size)])[::-1] [:topk]
      len_list_str = [str(i) for i in len_list]
      lens = '|'.join(len_list_str)
      print(lens)

  def compute_lp_scores (self,all_embeds,query_nodes,candidate_list, k, use_tensor=False):
    """
      return both AP/MAP for given candidate_list    
    """
    agglo_k = len(query_nodes)*k
    time_dict = {}
    time_dict['start_score_computation'] = time.time()
    #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    #a,b = zip(*candidate_list)
    #self_tensors = torch.index_select(all_embeds,dim=0,index=cudavar(self.av,torch.tensor(a)))
    #nbr_tensors  = torch.index_select(all_embeds,dim=0,index=cudavar(self.av,torch.tensor(b)))
    #scores   = cos(self_tensors,nbr_tensors).tolist()
    #cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    #scores = []
    #for (a,b) in candidate_list:
    #  scores.append( cos(all_embeds[a],all_embeds[b]))
    if use_tensor:
      cos = nn.CosineSimilarity(dim=1, eps=1e-6)
      a,b = zip(*candidate_list)
      self_tensors = torch.index_select(all_embeds,dim=0,index=cudavar(self.av,torch.tensor(a)))
      nbr_tensors  = torch.index_select(all_embeds,dim=0,index=cudavar(self.av,torch.tensor(b)))
      scores   = cos(self_tensors,nbr_tensors).tolist()
    else:  
      cos = nn.CosineSimilarity(dim=0, eps=1e-6)
      scores = []
      for (a,b) in candidate_list:
        scores.append( cos(all_embeds[a],all_embeds[b]))
      scores = torch.stack(scores).tolist()  
      
    time_dict['end_score_computation'] = time.time()
    time_dict['start_heap_procedure'] = time.time()
    score_heap = []
    heap_size = 0
    qnode_heap_dict = {}
    qnode_heap_size_dict = {}
    for node in query_nodes:
      qnode_heap_dict[node] = []
      qnode_heap_size_dict[node ]= 0

    for i in range(len(candidate_list)):
      if heap_size<agglo_k: 
        heap_size = heap_size+1
        heapq.heappush(score_heap,(scores[i],candidate_list[i]))
      else:
        heapq.heappushpop(score_heap,(scores[i],candidate_list[i]))
      for node in candidate_list[i]: 
        if node in query_nodes: 
          if qnode_heap_size_dict[node]<k:
            qnode_heap_size_dict[node] = qnode_heap_size_dict[node]+1
            heapq.heappush(qnode_heap_dict[node], (scores[i],candidate_list[i]))
          else:
            heapq.heappushpop(qnode_heap_dict[node], (scores[i],candidate_list[i]))  

    time_dict['end_heap_procedure'] = time.time() 
    scores,predicted_edges =  list(zip (*score_heap))
    all_scores = list(scores)
    all_labels =np.array([self.candidate_set[a][b] for (a,b) in list(list(predicted_edges))])
    all_labels[all_labels==-1] = 0
    if np.all(all_labels == 1):
      ap_score = 1
    elif np.all(all_labels == 0):
      ap_score = 0
    else:
      ap_score   = average_precision_score(all_labels, all_scores)   
      
    ndcg = ndcg_score([all_labels], [all_scores])#,k=agglo_k)   
    
    ap_score_agglo = ap_score
    ndcg_score_agglo = ndcg

    all_qnode_ap = []
    all_qnode_ndcg = []
    for qnode in query_nodes :
      if qnode_heap_size_dict[qnode]==0:
        continue
      scores,predicted_edges =  list(zip (*qnode_heap_dict[qnode]))
      all_scores = list(scores)
      all_labels =np.array([self.candidate_set[a][b] for (a,b) in list(list(predicted_edges))])
      all_labels[all_labels==-1] = 0
      if np.all(all_labels == 1):
        ap_score = 1
        ndcg=1
      elif np.all(all_labels == 0):
        ap_score = 0
        ndcg=0
      else:
        ap_score   = average_precision_score(all_labels, all_scores)
        ndcg   = ndcg_score([all_labels], [all_scores])
      all_qnode_ap.append(ap_score)  
      all_qnode_ndcg.append(ndcg)  

    return ap_score_agglo, ndcg_score_agglo, np.mean(all_qnode_ap), np.mean(all_qnode_ndcg),time_dict  


  def get_hash_lp_scores(self,all_embeds,all_hashcodes,query_nodes, list_edges, list_non_edges, k=10,no_bucket=False): 
    """
      given k, find the top k closest node pairs from the buckets
      loop over al hash_tables: 
        loop over all buckets: 
          nC2 in nodes in each bucket, compute cosine similarity and update min heap
      return auc/ap of pairs in min heap    
    """
    start = time.time()
    self.init_hash_code_mat(all_hashcodes) 
    self.init_candidate_set(list_edges,list_non_edges)
    self.bucketify()
    bucket_node_pair_count=0
    start_candidate_list_gen = time.time()
    if no_bucket: 
      candidate_list = list_edges + list_non_edges
    else:
      candidate_list = []
      for table_id in range(self.num_hash_tables): 
        for bucket_id in range(2**self.subset_size): 
          node_list = self.all_hash_tables[table_id][bucket_id]
          node_pairs = list(combinations(node_list,2))
          candidate_list.extend(node_pairs)
        
      #remove duplicates from candidate_list
      candidate_list = list(set(candidate_list))
      bucket_node_pair_count = len(candidate_list)
      #filter list_test_edges and list_test_non_edges
      candidate_list =list(filter(lambda x: self.candidate_set[x[0]][x[1]]!=0,candidate_list))
    end_candidate_list_gen = time.time()
    new_q = []
    for qnode in query_nodes: 
      qnode_edges = list(filter(lambda x: x[0]==qnode or x[1]==qnode, list_edges))
      qnode_non_edges = list(filter(lambda x: x[0]==qnode or x[1]==qnode, list_non_edges))
      if len(qnode_edges) ==0 or len(qnode_non_edges)==0:
          continue
      new_q.append(qnode)

    ap_score,ndcg,map_score,mndcg, time_dict = self.compute_lp_scores (all_embeds,new_q,candidate_list, k)
    time_dict['start_candidate_list_gen'] = start_candidate_list_gen  
    time_dict['end_candidate_list_gen'] = end_candidate_list_gen  
    return bucket_node_pair_count, len(candidate_list),  ap_score,ndcg,map_score, mndcg,time.time()-start,time_dict
