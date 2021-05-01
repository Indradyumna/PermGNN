import numpy as np
import networkx as nx
from random import sample
from collections import defaultdict
from common import logger
import copy
import os

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, average_precision_score

from common import logger, set_log
from minmax.utils import cudavar

def generate_query_nodes(gr,threshold=0): 
  gr_nx = gr.get_nx_graph()
  #self loops throws off triangle detection. Hence removed
  gr_nx.remove_edges_from(nx.selfloop_edges(gr_nx))
  gr_adj_mat = nx.to_numpy_matrix(gr_nx)
  #(i,j of A^3 is no of 3 len paths between i,j)
  three_path_len = gr_adj_mat*gr_adj_mat*gr_adj_mat
  #3 len path ending on start node are triangles.
  #Each triangle counted twice - cw and anti-cw
  triangle_count = np.diag(three_path_len)
  #2*threshold because all triangle counts are even numbered
  query_nodes = list(filter(lambda x: triangle_count[x]>2*threshold, range(len(triangle_count))))
  return query_nodes

def populate_query_set(gr,query_nodes): 
  G = gr.get_nx_graph()
  #remove nodes from their own neighbourhood set
  G.remove_edges_from(nx.selfloop_edges(G))

  #code to limit non-edge set to 2-path reachable nodes
  G_adj_mat = nx.to_numpy_matrix(G)
  #A^2+A has non-zero entries for 1-len and 2-len reachable nodes for adjacency matrix A
  G_mod = nx.from_numpy_matrix(G_adj_mat*G_adj_mat + G_adj_mat)

  query_full_set = {}
  for q_node in query_nodes:
    query_full_set[q_node] = {}
    query_full_set[q_node]['Nbr']    = list(nx.neighbors(G,q_node))
    query_full_set[q_node]['NonNbr'] =  list(set(nx.non_neighbors(G,q_node))\
                                             -set(nx.non_neighbors(G_mod,q_node)))

  #assert 'Nbr' and 'NonNbr' for all q_nodes are complements wrt 2-len neighbourhood and neither contain q_node  
  assert( all([ \
            (set(query_full_set[q_node]['Nbr']).intersection(query_full_set[q_node]['NonNbr']) == set() ) \
                and (len(query_full_set[q_node]['Nbr']) + len(query_full_set[q_node]['NonNbr'])==len(list(nx.neighbors(G_mod,q_node)))-1) \
                and (q_node not in set(query_full_set[q_node]['Nbr']).union(query_full_set[q_node]['NonNbr'])) \
                    for q_node in query_nodes]) \
            ==True)
  return query_full_set  

def split_query_set(split_frac,query_nodes,query_full_set): 
  query_training_set = {}
  query_test_set = {}
  for q_node in query_nodes:
    query_training_set[q_node] = {}
    query_test_set[q_node]     = {}
    for key in ['Nbr','NonNbr'] :
      test_len = int(split_frac * len(query_full_set[q_node][key]))
      query_test_set[q_node][key] = sample(query_full_set[q_node][key],test_len)
      query_training_set[q_node][key] = list(set(query_full_set[q_node][key]) \
                                           - set(query_test_set[q_node][key]))

  return  query_training_set, query_test_set      

def generate_edge_lists(gr,query_nodes,query_training_set, query_test_set):

  list_test_edges = []
  list_test_non_edges = [] 
  list_training_edges = []
  list_training_non_edges = [] 
  for q_node in query_nodes:
    for x in query_test_set[q_node]['Nbr']    : list_test_edges.append((q_node, x))
    for x in query_test_set[q_node]['NonNbr'] : list_test_non_edges.append((q_node, x))
        
  #if (x,y) has been sampled as test edge set, (y,x) should not be in training edge set
  #We've decided to remove overlap from training
  hashM = np.zeros((gr.get_num_nodes(),gr.get_num_nodes()))
  for q_node in query_nodes: 
    for x in query_test_set[q_node]['Nbr'] : hashM[q_node][x]=1 
  for q_node in query_nodes: 
    templist = copy.deepcopy(query_training_set[q_node]['Nbr'] )
    for x in templist : 
      if hashM[x][q_node]!=1 :   list_training_edges.append((q_node, x))
      else :
        query_training_set[q_node]['Nbr'].remove(x)   
  
  hashM = np.zeros((gr.get_num_nodes(),gr.get_num_nodes()))
  for q_node in query_nodes: 
    for x in query_test_set[q_node]['NonNbr'] : hashM[q_node][x]=1 
  for q_node in query_nodes: 
    templist = copy.deepcopy(query_training_set[q_node]['NonNbr'])
    for x in  templist: 
      if hashM[x][q_node]!=1 :   list_training_non_edges.append((q_node, x))
      else : query_training_set[q_node]['NonNbr'].remove(x) 

  return list_training_edges,list_training_non_edges,list_test_edges,list_test_non_edges  

def prep_permgnn_graph(av,gr, query_nodes,list_training_edges,list_training_non_edges,list_val_edges,list_test_edges,list_val_non_edges,list_test_non_edges): 
  for (q_node,x) in list_test_edges: 
    #NOTE: discard instead of remove because (a,b) and (b,a) may both be in list
    gr.adjacency_list[q_node].discard(x)
    gr.adjacency_list[x].discard(q_node)
  for (q_node,x) in list_val_edges: 
    #NOTE: discard instead of remove because (a,b) and (b,a) may both be in list
    gr.adjacency_list[q_node].discard(x)
    gr.adjacency_list[x].discard(q_node)
  gr.update_query_node_info_tr(list_val_edges, list_val_non_edges, list_test_edges, list_test_non_edges)  
  #below 4 lines are not necessary for functionality 
  #included here to check consistency of edge list generated from updated
  #adjacency_matrix above and training_neighbour_set below
  set_of_all_nodes = set(range(gr.get_num_nodes()))
  list_of_neighbours=list(map(lambda x:gr.adjacency_list[x],set_of_all_nodes))
  list_of_non_neighbours = list(map(lambda x:set_of_all_nodes-x,list_of_neighbours))
  list_of_neighbours = list(x1 - set([x2]) for (x1, x2) in zip(list_of_neighbours,set_of_all_nodes))
  
  training_neighbour_set = defaultdict(set) 
  for(q_node,x) in list_training_edges:
    training_neighbour_set[q_node].add(x)
    training_neighbour_set[x].add(q_node)
  #below assert confirms overall behavior till now is correct
  assert(all(\
           [(list_of_neighbours[node]==training_neighbour_set[node]) for node in query_nodes])\
       ==True)
  #permGnnGraph assumes edge and non_edge sets to be complementary
  #that is not true post splitting of non_edge set into train:test
  training_non_neighbour_set = defaultdict(set)
  for(q_node,x) in list_training_non_edges:
    training_non_neighbour_set[q_node].add(x)
    training_non_neighbour_set[x].add(q_node)

  gr.max_node_outdegree =max([len(gr.adjacency_list[x]) for x in range(gr.num_nodes)]) - 1
 
def compute_scores_from_embeds(av,all_embeds,query_nodes,list_test_edges,list_test_non_edges):
  cos = nn.CosineSimilarity(dim=1, eps=1e-6)
  #per qnode
  #all_qnode_auc = [] 
  all_qnode_ap = []
  all_qnode_rr = []
  #all_qnode_ndcg = []
  for qnode in query_nodes : 
    qnode_edges = list(filter(lambda x: x[0]==qnode or x[1]==qnode, list_test_edges))
    qnode_non_edges = list(filter(lambda x: x[0]==qnode or x[1]==qnode, list_test_non_edges))
    if len(qnode_edges)==0 or len(qnode_non_edges)==0: 
      continue
    a,b = zip(*qnode_edges)
    self_tensors = torch.index_select(all_embeds,dim=0,index=cudavar(av,torch.tensor(a)))
    nbr_tensors  = torch.index_select(all_embeds,dim=0,index=cudavar(av,torch.tensor(b)))
    pos_scores   = cos(self_tensors,nbr_tensors)

    a,b = zip(*qnode_non_edges)
    self_tensors = torch.index_select(all_embeds,dim=0,index=cudavar(av,torch.tensor(a)))
    nbr_tensors  = torch.index_select(all_embeds,dim=0,index=cudavar(av,torch.tensor(b)))
    neg_scores   = cos(self_tensors,nbr_tensors)

    if av.has_cuda and av.want_cuda:
      all_scores = torch.cat((pos_scores,neg_scores)).cpu().numpy()
    else:
      all_scores = torch.cat((pos_scores,neg_scores)).numpy()

    all_labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    auc_score  = roc_auc_score(all_labels, all_scores)
    ap_score   = average_precision_score(all_labels, all_scores)
    #ndcg       = ndcg_score([all_labels],[all_scores])

    so = np.argsort(all_scores)[::-1]
    labels_rearranged = all_labels[so]
    rr_score = 1/(labels_rearranged.tolist().index(1)+1)
    
    #all_qnode_auc.append(auc_score)
    all_qnode_ap.append(ap_score)
    all_qnode_rr.append(rr_score)
    #all_qnode_ndcg.append(ndcg)
  #agglo
  pos_scores = []
  neg_scores = []

  a,b = zip(*list_test_edges)
  self_tensors = torch.index_select(all_embeds,dim=0,index=cudavar(av,torch.tensor(a)))
  nbr_tensors  = torch.index_select(all_embeds,dim=0,index=cudavar(av,torch.tensor(b)))
  pos_scores   = cos(self_tensors,nbr_tensors)

  a,b = zip(*list_test_non_edges)
  self_tensors = torch.index_select(all_embeds,dim=0,index=cudavar(av,torch.tensor(a)))
  nbr_tensors  = torch.index_select(all_embeds,dim=0,index=cudavar(av,torch.tensor(b)))
  neg_scores   = cos(self_tensors,nbr_tensors)

  if av.has_cuda and av.want_cuda:
    all_scores = torch.cat((pos_scores,neg_scores)).cpu().numpy()
  else:
    all_scores = torch.cat((pos_scores,neg_scores)).numpy()

  all_labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
  auc_score  = roc_auc_score(all_labels, all_scores)
  ap_score   = average_precision_score(all_labels, all_scores)
  #ndcg       = ndcg_score([all_labels],[all_scores])
  
  #so = np.argsort(all_scores)[::-1]
  #labels_rearranged = all_labels[so]
  #rr_score = 1/(labels_rearranged.tolist().index(1)+1)

  return auc_score, ap_score, np.mean(all_qnode_ap), np.mean(all_qnode_rr)

def compute_scores(av,permGNN,query_nodes,list_test_edges,list_test_non_edges):
  all_nodes = list(range(permGNN.gr.get_num_nodes()))
  all_embeds = cudavar(av,torch.tensor([]))
  #batch and send nodes to avoid memory limit crash for larger graphs
  for i in range(0,permGNN.gr.get_num_nodes(),av.BATCH_SIZE) : 
    batch_nodes = all_nodes[i:i+av.BATCH_SIZE]
    all_embeds = torch.cat((all_embeds,permGNN.forward(batch_nodes).data),dim=0)
  return compute_scores_from_embeds(av,all_embeds,query_nodes,list_test_edges,list_test_non_edges) 

def get_lp_scores(av,permGNN,query_nodes,list_test_edges,list_test_non_edges):
  if av.SCORE == "MAP": 
    _,_,map_score,_ = compute_scores(av,permGNN,query_nodes,list_test_edges,list_test_non_edges)
    return map_score
  elif av.SCORE == "AUC_AP":
    auc_score,ap_score,_,_ = compute_scores(av,permGNN,query_nodes,list_test_edges,list_test_non_edges)
    return auc_score,ap_score
  else:
    raise NotImplementedError()

def fetch_lp_data_split(av,gr): 
  if os.path.exists(os.path.join(av.DIR_PATH, "data/savedDataSplit",\
                                 av.DATASET_NAME + "_tfrac_" + \
                                 str(av.TEST_FRAC) + "_vfrac_" + \
                                 str(av.VAL_FRAC))):
    query_nodes, \
    list_training_edges, \
    list_training_non_edges, \
    list_test_edges, \
    list_test_non_edges, \
    list_val_edges, \
    list_val_non_edges = load_data_split(av)
  else :   
    query_nodes = generate_query_nodes(gr)

  
    query_full_set =  populate_query_set(gr,query_nodes)

    query_training_set, query_test_set =  split_query_set(av.TEST_FRAC,query_nodes,query_full_set)
    list_training_edges, list_training_non_edges, list_test_edges, list_test_non_edges \
           = generate_edge_lists(gr,query_nodes,query_training_set, query_test_set)

    if av.VAL_FRAC==0:
      list_val_edges = []
      list_val_non_edges = []
    else :
      query_training_only_set, query_val_set =  split_query_set(av.VAL_FRAC,query_nodes,query_training_set)
      list_training_edges, list_training_non_edges, list_val_edges, list_val_non_edges \
           = generate_edge_lists(gr,query_nodes,query_training_only_set, query_val_set)
    
    save_data_split(av,query_nodes, list_training_edges, list_training_non_edges, \
                       list_test_edges, list_test_non_edges, list_val_edges, list_val_non_edges )       
  
  logger.info("Dataset \"%s\" has %d nodes of which %d are query nodes",  \
        av.DATASET_NAME,gr.get_num_nodes(),len(query_nodes))  
  logger.info("Intended test split_frac was %f \
         Actual split frac for edges was %f\
         and non-edges was %f",\
        av.TEST_FRAC,\
               len(list_test_edges)/(len(list_test_edges)+len(list_training_edges)),\
               len(list_test_non_edges)/(len(list_test_non_edges)+len(list_training_non_edges))) 
  if av.VAL_FRAC==0:
    logger.info("Intended val split_frac was %f",av.VAL_FRAC)
  else:    
    logger.info("Intended val split_frac was %f \
         Actual split frac for edges was %f\
         and non-edges was %f",\
         av.VAL_FRAC,\
               len(list_val_edges)/(len(list_val_edges)+len(list_training_edges)),\
               len(list_val_non_edges)/(len(list_val_non_edges)+len(list_training_non_edges)))

  return query_nodes, \
    list_training_edges, \
    list_training_non_edges, \
    list_test_edges, \
    list_test_non_edges, \
    list_val_edges, \
    list_val_non_edges 

def save_data_split(av,query_nodes,list_training_edges,list_training_non_edges,list_test_edges,list_test_non_edges,list_val_edges,list_val_non_edges):
  save_dir = os.path.join(av.DIR_PATH, "data/savedDataSplit")
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  name = av.DATASET_NAME
  save_path = os.path.join(save_dir, name) + "_tfrac_" + str(av.TEST_FRAC) + "_vfrac_" + str(av.VAL_FRAC)
  logger.info("saving data split to %s",save_path)
  output = open(save_path, mode="wb")
  torch.save({
            'query_nodes': query_nodes,
            'list_training_edges': list_training_edges,
            'list_training_non_edges': list_training_non_edges,
            'list_test_edges': list_test_edges,
            'list_test_non_edges': list_test_non_edges,
            'list_val_edges': list_val_edges,
            'list_val_non_edges': list_val_non_edges,
            }, output)
  output.close() 

def load_data_split(av):
  load_dir = os.path.join(av.DIR_PATH, "data/savedDataSplit")
  if not os.path.isdir(load_dir):
    raise Exception('{} does not exist'.format(load_dir))
  name = av.DATASET_NAME    
  load_path = os.path.join(load_dir, name) + "_tfrac_" + str(av.TEST_FRAC) + "_vfrac_" + str(av.VAL_FRAC)
  logger.info("loading data split from %s",load_path)
  checkpoint = torch.load(load_path)
  return checkpoint['query_nodes'],checkpoint['list_training_edges'],\
    checkpoint['list_training_non_edges'],checkpoint['list_test_edges'],checkpoint['list_test_non_edges'],\
    checkpoint['list_val_edges'],checkpoint['list_val_non_edges']    
