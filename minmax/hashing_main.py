import argparse
import random
import time
import pickle
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from common import logger, set_log
from minmax.graphs import PermGnnGraph,LinqsUmdDocumentGraph,TwitterGraph,ABGraph
from minmax.utils import cudavar
from minmax.linkpred_utils import prep_permgnn_graph, fetch_lp_data_split 
from minmax.linkpred_main import fetch_permgnn_embeddings 
from minmax.earlystopping import EarlyStoppingModule
from minmax.locality_sensitive_hashing import LSH

class HashCodeGenerator(nn.Module):
  """
    Fetch embeddings output from model 
    feed into NN(Linear+tanh)
    Compute loss on hashcode 
  """
  def __init__(self, av,gr: PermGnnGraph):
    super(HashCodeGenerator, self).__init__()
    self.av = av
    self.gr = gr
    self.all_embeddings = nn.Embedding(self.gr.get_num_nodes() ,self.av.EMBEDDING_DIM)
    self.non_nbr_mat = cudavar(self.av,torch.zeros(self.gr.get_num_nodes(),self.gr.get_num_nodes()))
    #Reusing PERM_NETWORK_LATENT_DIM here because why not \()/
    self.latent_dim = self.av.PERM_NETWORK_LATENT_DIM
    self.hash_linear1 = nn.Linear(self.av.EMBEDDING_DIM, self.av.HASHCODE_DIM)
    self.hash_tanh1  = nn.Tanh()
    nn.init.normal_(self.hash_linear1.weight)

  def init_embeddings(self,embeddings): 
    self.all_embeddings.weight =  nn.Parameter(embeddings,requires_grad=False)
  
  def init_non_nbr_mat(self,list_training_edges): 
    for (a,b) in list_training_edges:
      self.non_nbr_mat[a][b] = 1 
    z = cudavar(self.av,torch.zeros(self.gr.get_num_nodes(),self.gr.get_num_nodes()))     
    o = cudavar(self.av,torch.ones(self.gr.get_num_nodes(),self.gr.get_num_nodes()))
    reverse = torch.where(self.non_nbr_mat==0,o,z)
    self.non_nbr_mat = reverse


  def forward(self, nodes): 
    node_embeddings = self.all_embeddings(cudavar(av,torch.LongTensor(nodes)))
    node_hashcodes = self.hash_tanh1(self.hash_linear1(node_embeddings))
    return node_hashcodes


  def computeLoss(self, nodes):
    """
      :param   nodes  : batch of node ids from range 0..NUM_NODES
      :return  loss   : Hinge ranking loss
    """
    loss1 = loss2 = loss3 = 0
    all_hashcodes = self.forward(nodes)
    num_nodes = len(nodes)

    for i in range(len(nodes)):
      selfcode = all_hashcodes[i]
      
      loss1 = loss1 +  torch.abs(torch.sum(selfcode))
      loss2 = loss2 +  torch.norm(torch.abs(selfcode)-1,p=1)

    indices = cudavar(av,torch.tensor(nodes))
    non_nbrs = torch.index_select(torch.index_select(self.non_nbr_mat,0,indices),1,indices)   
    similarity_mat =  torch.mul(torch.abs(torch.mm(all_hashcodes,torch.transpose(all_hashcodes,0,1))),non_nbrs)

    loss3 = torch.sum(similarity_mat) - torch.sum(torch.diagonal(similarity_mat))

    return loss1,loss2,loss3,num_nodes


def run_lsh(av,gr: PermGnnGraph):

  all_hashing_info_dict = {}
  all_embeds = fetch_permgnn_embeddings(av,gr)
  
  pickle_fp = av.DIR_PATH+ "/data/hashcodePickles/"+av.TASK+"_"+av.DATASET_NAME+"_tfrac_"+str(av.TEST_FRAC)+"_vfrac_"+str(av.VAL_FRAC) +"_L1_" + str(av.LAMBDA1) + "_L2_" + str(av.LAMBDA2)+ "_hashcode_mat.pkl"
  with open(pickle_fp, 'rb') as f:
    all_hashcodes = pickle.load(f)  
  all_hashcodes = all_hashcodes.float()  
  
  pickle_fp = av.DIR_PATH+ "/data/hashcodePickles/"+av.TASK+"_gaussian_"+av.DATASET_NAME+"_tfrac_"+str(av.TEST_FRAC)+"_vfrac_"+str(av.VAL_FRAC) + "_hashcode_mat.pkl"
  with open(pickle_fp, 'rb') as f:
    all_hashcodes_gaussian = pickle.load(f)  
  all_hashcodes_gaussian = all_hashcodes_gaussian.float()  
  
  query_nodes, \
    list_training_edges, \
    list_training_non_edges, \
    list_test_edges, \
    list_test_non_edges, \
    list_val_edges, \
    list_val_non_edges = fetch_lp_data_split(av,gr)

  prep_permgnn_graph(av,gr,query_nodes,list_training_edges,list_training_non_edges,list_val_edges,list_test_edges,list_val_non_edges,list_test_non_edges)
  #TODO:input hparam suport for d and k
  d=8
  k=10

  lsh = LSH(av,gr,10,d)
  _,_, ap,ndcg,map_sc,mndcg, time_total, time_dict = lsh.get_hash_lp_scores(all_embeds,all_hashcodes,query_nodes, list_test_edges, list_test_non_edges,k,True)
  node_pair_count = (gr.get_num_nodes()*(gr.get_num_nodes()-1))/2
  test_pair_count = len(list_test_edges) + len(list_test_non_edges)
  all_hashing_info_dict['nohash']= [node_pair_count, test_pair_count,ap,ndcg,map_sc,mndcg,time_total,time_dict['end_score_computation']-time_dict['start_score_computation'],time_dict['end_heap_procedure']-time_dict['start_heap_procedure'],time_dict['end_candidate_list_gen']-time_dict['start_candidate_list_gen']]
  

  lsh.init_candidate_set(list_test_edges,list_test_non_edges)
  
  lsh.init_hash_code_mat(all_hashcodes) 
  lsh.bucketify()
  len_test, len_candidate, ap,ndcg,map_sc,mndcg, time_total, time_dict = lsh.get_hash_lp_scores(all_embeds,all_hashcodes,query_nodes, list_test_edges, list_test_non_edges,k,False)
  all_hashing_info_dict['trained'] = [len_test, len_candidate,ap,ndcg,map_sc,mndcg,time_total,time_dict['end_score_computation']-time_dict['start_score_computation'],time_dict['end_heap_procedure']-time_dict['start_heap_procedure'],time_dict['end_candidate_list_gen']-time_dict['start_candidate_list_gen']]

  lsh.init_hash_code_mat(all_hashcodes_gaussian) 
  lsh.bucketify()
  len_test, len_candidate, ap,ndcg,map_sc,mndcg, time_total, time_dict = lsh.get_hash_lp_scores(all_embeds,all_hashcodes_gaussian,query_nodes, list_test_edges, list_test_non_edges,k,False)
  all_hashing_info_dict['gaussian'] = [len_test, len_candidate,ap,ndcg,map_sc,mndcg,time_total,time_dict['end_score_computation']-time_dict['start_score_computation'],time_dict['end_heap_procedure']-time_dict['start_heap_procedure'],time_dict['end_candidate_list_gen']-time_dict['start_candidate_list_gen']]
  
  #print in file  
  fp =  "data/logDir/hashing_info_"+av.DATASET_NAME+"_l1_"+str(av.LAMBDA1)+"_l2_"+str(av.LAMBDA2)+".txt"
  f = open(fp,'w+')
  logger.info("Writing results to file %s",fp)
  for ver in ['nohash','trained','gaussian']:
    info = all_hashing_info_dict[ver]
    f.write("version: {} node_pair_count: {} test_pair_count: {}  ap: {} ndcg: {} map: {} mndcg: {} time: {} time_candidate_list_gen: {} time_scoring: {} time_heaping: {}".format(ver, info[0],info[1],info[2],info[3],info[4],info[5],info[6],info[9],info[7], info[8]))
    f.write('\n')
  

def run_graph_lp_hash(av,gr: PermGnnGraph):
  pickle_fp = av.DIR_PATH + "/data/hashcodePickles/"+av.TASK+"_"+av.DATASET_NAME+"_tfrac_"+str(av.TEST_FRAC)+"_vfrac_"+str(av.VAL_FRAC) +"_L1_" + str(av.LAMBDA1) + "_L2_" + str(av.LAMBDA2)+ "_hashcode_mat.pkl"
  if not os.path.exists(pickle_fp):
    #if av.has_cuda:
    #  torch.cuda.reset_max_memory_allocated(0)
    #fetch permGNN embeddings
    device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    query_nodes, \
      list_training_edges, \
      list_training_non_edges, \
      list_test_edges, \
      list_test_non_edges, \
      list_val_edges, \
      list_val_non_edges = fetch_lp_data_split(av,gr)

    prep_permgnn_graph(av,gr,query_nodes,list_training_edges,list_training_non_edges,list_val_edges,list_test_edges,list_val_non_edges,list_test_non_edges) 
    all_embeds = fetch_permgnn_embeddings(av,gr)

    hashCodeGenerator = HashCodeGenerator(av,gr).to(device)
    hashCodeGenerator.init_embeddings(all_embeds)
    hashCodeGenerator.init_non_nbr_mat(list_training_edges)
    
    es = EarlyStoppingModule(av,50,0.001)

    optimizerFunc = torch.optim.SGD(hashCodeGenerator.parameters(), lr=av.LEARNING_RATE_FUNC)
    nodes = list(range(gr.get_num_nodes()))
    epoch = 0
    #if VAL_FRAC is 0, we train model for NUM_EPOCHS
    #else we train model till early stopping criteria is met
    while av.VAL_FRAC!=0 or epoch<av.NUM_EPOCHS:
      random.shuffle(nodes)
      start_time = time.time()
      totalEpochLoss =0 
      for i in range(0, gr.get_num_nodes(), av.BATCH_SIZE):
        nodes_batch = nodes[i:i+av.BATCH_SIZE]
        hashCodeGenerator.zero_grad()
        loss1,loss2,loss3,num_nodes = hashCodeGenerator.computeLoss(nodes_batch)
        totalLoss = (av.LAMBDA1/num_nodes)*loss1 + (av.LAMBDA2/num_nodes)*loss2 + ((1-(av.LAMBDA1+av.LAMBDA2))/(num_nodes**2))*loss3
        totalLoss.backward()
        optimizerFunc.step()
        totalEpochLoss = totalEpochLoss + totalLoss.item()   
      end_time = time.time()
      logger.info("Epoch: %d totalEpochLoss: %f time: %.2f", epoch,totalEpochLoss, end_time-start_time)
      if av.VAL_FRAC!=0:
        if es.check([-totalEpochLoss],hashCodeGenerator,epoch):
          break
      epoch+=1
    if av.has_cuda:
      logger.info("Max gpu memory used: %.6f ",torch.cuda.max_memory_allocated(device=0)/(1024**3))
    
    #generate and dump hashcode  pickles
    all_nodes = list(range(gr.get_num_nodes()))
    all_hashcodes = cudavar(av,torch.tensor([]))
    for i in range(0,gr.get_num_nodes(),av.BATCH_SIZE) :
      batch_nodes = all_nodes[i:i+av.BATCH_SIZE]
      all_hashcodes = torch.cat((all_hashcodes,hashCodeGenerator.forward(batch_nodes).data),dim=0)
    logger.info("Dumping trained hashcode pickle at %s",pickle_fp)
    with open(pickle_fp, 'wb') as f:
      pickle.dump(all_hashcodes, f)

def run_graph_lp_hash_gaussian(av,gr: PermGnnGraph):
  pickle_fp = av.DIR_PATH+ "/data/hashcodePickles/"+av.TASK+"_gaussian_"+av.DATASET_NAME+"_tfrac_"+str(av.TEST_FRAC)+"_vfrac_"+str(av.VAL_FRAC) + "_hashcode_mat.pkl"
  if not os.path.exists(pickle_fp):
    #fetch permGNN embeddings
    device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    all_embeds = fetch_permgnn_embeddings(av,gr)
    fp = av.DIR_PATH + "/data/hashcodePickles/gauss_hplanes_D_16.pkl"
    hplanes = pickle.load(open(fp,'rb'))
    projections = all_embeds.cpu().numpy()@np.transpose(hplanes)
    hcode = np.sign(projections)
    all_hashcodes = cudavar(av,torch.tensor(hcode))
    logger.info("Dumping gaussian hashcode pickle at %s",pickle_fp)
    with open(pickle_fp, 'wb') as f:
      pickle.dump(all_hashcodes, f)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--logpath",                 type=str,   default="data/logDir/logfile",help="/path/to/log")
  ap.add_argument("--want_cuda",               type=bool,  default=True)
  ap.add_argument("--has_cuda",                type=bool,  default=torch.cuda.is_available())
  ap.add_argument("--PERM_NETWORK_LATENT_DIM", type=int,   default=16)
  ap.add_argument("--LSTM_HIDDEN_DIM",         type=int,   default=32)
  ap.add_argument("--EMBEDDING_DIM",           type=int,   default=16)
  ap.add_argument("--HASHCODE_DIM",            type=int,   default=16)
  ap.add_argument("--NUM_EPOCHS",              type=int,   default=200)
  ap.add_argument("--BATCH_SIZE",              type=int,   default=128)
  ap.add_argument("--MARGIN",                  type=float, default=.01)
  ap.add_argument("--LEARNING_RATE_FUNC",      type=float, default=0.05)
  ap.add_argument("--DIR_PATH",                type=str,   default=".",help="path/to/datasets")
  ap.add_argument("--DATASET_NAME",            type=str,   default="Twitter_3", help="cora/citeseer/Twitter_3/Gplus_1/PB")
  ap.add_argument("--TEST_FRAC",               type=float, default=0.2)
  ap.add_argument("--VAL_FRAC",                type=float, default=0.1)
  ap.add_argument("--LAMBDA1",                 type=float, default=0.01)
  ap.add_argument("--LAMBDA2",                 type=float, default=0.01)
  ap.add_argument("--TOP_K",                   type=int,   default=0)
  ap.add_argument("--TASK",                    type=str,   default="LP_hash")

  av = ap.parse_args()

  av.logpath = av.logpath+"_"+av.TASK+"_"+av.DATASET_NAME+"_tfrac_"+str(av.TEST_FRAC)+"_vfrac_"+str(av.VAL_FRAC)+"_L1_" + str(av.LAMBDA1) + "_L2_" + str(av.LAMBDA2)
  set_log(av)
  logger.info("Command line")  
  logger.info('\n'.join(sys.argv[:]))  
  if av.DATASET_NAME in ['AB100','AB1k','AB5k','AB10k','AB20k','AB50k']:
    av.VAL_FRAC=0.0
    gr = ABGraph(av)
  elif av.DATASET_NAME in ['Twitter_3','Gplus_1','PB']:
    gr = TwitterGraph(av)
  else:
    gr = LinqsUmdDocumentGraph(av)
  run_graph_lp_hash(av,gr)
  run_graph_lp_hash_gaussian(av,gr)
  run_lsh(av,gr)
