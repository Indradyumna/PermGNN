import numpy as np
import networkx as nx
from collections import defaultdict
from common import logger
import pickle 
import os

class PermGnnGraph(object):
  """Empty graph."""
  def __init__(self):
    """
      node_features      : np.array of shape (num_nodes, num_features)
      adjacency_lists    : defaultdict(set) with key:value --> node id:set of neighbour node ids
                           node ids are always in range 0..num_nodes
                           node id itself is member of neighbourhood set.
                           Thus set sizes are in range 1..(max_node_outdegree+1)
    """
    self.adjacency_list     = defaultdict(set)
    self.node_features      = np.empty((0,0))
    self.num_nodes          = 0
    self.max_node_outdegree = 0
    """
      These three contain info regarding which node and which edge/non-edge
      computeLoss is run on during training
      query_node_list    : list of node ids which are subset of range (0,num_nodes)
                           default flow for permGnn embedding training has all nodes in this list
                           link_prediction flow will have query_nodes only
      query_node_nbr     : defaultdict(set) with key:value --> node id:set of neighbour node ids
                           default flow for permGnn embedding training has this same as adjacency_list
                           link_prediction flow will training_edge info for each query_node which is a subset of adjacency_list[query_node]
      query_node_non_nbr : defaultdict(set) with key:value --> node id:set of neighbour node ids
                           default flow for permgnn embedding training has this contain complement of adjacency_list
                           link_prediction flow will training_non_edge info for each query_node
    """

    self.query_node_list    = []
    self.query_node_nbr     = defaultdict(set)
    self.query_node_non_nbr = defaultdict(set)

  def get_num_nodes(self):
    return self.num_nodes

  def get_num_features(self):
    return self.node_features.shape[1]

  def get_max_node_outdegree(self):
    return self.max_node_outdegree

  def set_node_features(self):
    raise NotImplementedError()
  
  def set_adjacency_list(self):
    raise NotImplementedError()

  def set_query_node_info(self):
    raise NotImplementedError()
  
  def update_query_node_info_tr(self,list_val_edges, list_val_non_edges, list_test_edges, list_test_non_edges):
    raise NotImplementedError()

  def get_nx_graph(self):
    assert bool(self.adjacency_list)
    dict_of_lists = {x: list(self.adjacency_list[x]) for x in range(self.get_num_nodes())}
    return nx.from_dict_of_lists(dict_of_lists)

class LinqsUmdDocumentGraph(PermGnnGraph):
  """Source link - http://www.cs.umd.edu/~sen/lbc-proj/LBC.html
    ["cora","citeseer","washington","texas","cornell","wisconsin"]
    These are the Document Classification Datasets with following meta-pattern
    .content file has document descriptions as <paper_id> <word_attributes>+ <class_label>
    .cites file has citation graph as <ID of cited paper> <ID of citing paper>
    PS: Some issues in citeseer - .cites has paper ID not part of .content. These are skipped
    PPS: No class labels in toy dataset. Handled with an if statement
  """
  def __init__(self,av):
    """
      :param av      : args
    """
    PermGnnGraph.__init__(self)
    self.av           = av
    self.dataset_name = av.DATASET_NAME
    self.content_path = self.av.DIR_PATH+"/data/Datasets/"+ self.dataset_name + ".content"
    self.cites_path   = self.av.DIR_PATH+"/data/Datasets/"+ self.dataset_name + ".cites"
    self.num_nodes    = sum(1 for line in open(self.content_path))
    self.node_map     = {}
    self.set_node_features()
    self.set_adjacency_list()
    self.max_node_outdegree = max([len(self.adjacency_list[x]) for x in range(self.num_nodes)]) - 1
    self.set_query_node_info()
    assert len(self.adjacency_list) == self.node_features.shape[0]
    logger.info("num_nodes=%s num_features=%d max_node_outdegrees=%d",
                self.get_num_nodes(), self.get_num_features(),
                self.get_max_node_outdegree())

  def set_node_features(self):   
    #Class labels are handled separately
    num_feats = len(open(self.content_path).readline().rstrip().split()) - (1 if self.dataset_name=='toy' else 2)
    self.node_features = np.empty((self.num_nodes, num_feats))
    labels = np.empty((self.num_nodes,1), dtype=np.int64)
    label_map = {}
    with open(self.content_path) as fp:
      for i,line in enumerate(fp):
        info = line.strip().split()
        if self.dataset_name=="toy":
          self.node_features [i,:] = info[1:]
        else:
          self.node_features [i,:] = [float(x) for x in info[1:-1]]
        self.node_map[info[0]] = i
        if not info[-1] in label_map:
          label_map[info[-1]] = len(label_map)
        labels[i] = label_map[info[-1]]

  def set_adjacency_list(self): 
    with open(self.cites_path) as fp:
      for i,line in enumerate(fp):
        info = line.strip().split()
        #Problem in "citeseer.cites" There are 17 edges referring nodes, not mentioned in citeseer.content
        if (info[0] not in self.node_map or info[1] not in self.node_map) : 
          continue  
        node1 = self.node_map[info[0]]
        node2 = self.node_map[info[1]]
        self.adjacency_list[node1].add(node2)
        self.adjacency_list[node2].add(node1)
      #node itself is considered in neighbourhood set 
      for i in range(self.num_nodes):
        self.adjacency_list[i].add(i)

  def set_query_node_info(self):
    self.query_node_list    = list(range(self.get_num_nodes()))
    #list of neighbour sets are cached for lookup in computeLoss (This containes the node itself)
    self.query_node_nbr     = {x: self.adjacency_list[x] for x in self.query_node_list}
    #list of non-neighbour sets are cached for lookup in computeLoss
    self.query_node_non_nbr = {x: set(self.query_node_list) - self.adjacency_list[x] for x in self.query_node_list}
    #removing the node itself from the neighbourhood set
    self.query_node_nbr     = {x:self.query_node_nbr[x]-{x} for x in self.query_node_list}

  def update_query_node_info_tr(self,list_val_edges, list_val_non_edges, list_test_edges, list_test_non_edges):
    for n1,n2 in list_test_edges:
       self.query_node_nbr[n1].discard(n2)
       self.query_node_nbr[n2].discard(n1)
    for n1,n2 in list_val_edges:
       self.query_node_nbr[n1].discard(n2)
       self.query_node_nbr[n2].discard(n1) 
    for n1,n2 in list_test_non_edges:
       self.query_node_non_nbr[n1].discard(n2)
       self.query_node_non_nbr[n2].discard(n1)
    for n1,n2 in list_val_non_edges:
       self.query_node_non_nbr[n1].discard(n2)
       self.query_node_non_nbr[n2].discard(n1) 

class TwitterGraph(PermGnnGraph):
  """
    This dataset has only edge list information.
    Node features are absent - we use one-hot node id as initial features
    VVIMP: If one decides to proceed with structural features for each node
      (pagerank, common neighbours, adamic adar, preferential attachment etc.)
      be sure to do feature extraction after dataset edgelist split to train/val/test set
      Since node features are dependent on edgelist information.
    PS:This class can be reused for graphs other than Twitter as required later
  """
  def __init__(self,av):
    """
      :param av      : args
    """
    PermGnnGraph.__init__(self)
    self.av           = av
    self.dataset_name = av.DATASET_NAME
    self.content_path = self.av.DIR_PATH+"/data/Datasets/"+ self.dataset_name + "_node_features.pkl"
    self.cites_path   = self.av.DIR_PATH+"/data/Datasets/"+ self.dataset_name + ".txt"
    self.set_adjacency_list()
    self.set_node_features()
    self.set_query_node_info()
    self.max_node_outdegree = max([len(self.adjacency_list[x]) for x in range(self.num_nodes)]) - 1
    assert len(self.adjacency_list) == self.node_features.shape[0]

  def set_node_features(self):
    self.node_features = np.identity(self.get_num_nodes())

  def set_adjacency_list(self):
    gr = nx.convert_node_labels_to_integers(nx.read_edgelist(self.cites_path))
    #node itself is considered in neighbourhood set
    gr.add_edges_from(zip(nx.nodes(gr),nx.nodes(gr)))
    dict_of_lists = nx.to_dict_of_lists(gr)
    self.adjacency_list = {x: set(dict_of_lists[x]) for x in list(nx.nodes(gr))}
    self.num_nodes = gr.number_of_nodes()

  def set_query_node_info(self):
    self.query_node_list    = list(range(self.get_num_nodes()))
    #list of neighbour sets are cached for lookup in computeLoss (This containes the node itself)
    self.query_node_nbr     = {x: self.adjacency_list[x] for x in self.query_node_list}
    #list of non-neighbour sets are cached for lookup in computeLoss
    self.query_node_non_nbr = {x: set(self.query_node_list) - self.adjacency_list[x] for x in self.query_node_list}
    #removing the node itself from the neighbourhood set
    self.query_node_nbr     = {x:self.query_node_nbr[x]-{x} for x in self.query_node_list}
  
  def update_query_node_info_tr(self,list_val_edges, list_val_non_edges, list_test_edges, list_test_non_edges):
    for n1,n2 in list_test_edges:
       self.query_node_nbr[n1].discard(n2)
       self.query_node_nbr[n2].discard(n1)
    for n1,n2 in list_val_edges:
       self.query_node_nbr[n1].discard(n2)
       self.query_node_nbr[n2].discard(n1) 
    for n1,n2 in list_test_non_edges:
       self.query_node_non_nbr[n1].discard(n2)
       self.query_node_non_nbr[n2].discard(n1)
    for n1,n2 in list_val_non_edges:
       self.query_node_non_nbr[n1].discard(n2)
       self.query_node_non_nbr[n2].discard(n1) 



class ABGraph(PermGnnGraph):
  """
    This dataset has only edge list information.
    Node features are absent - we use one-hot node id as initial features
    VVIMP: If one decides to proceed with structural features for each node
      (pagerank, common neighbours, adamic adar, preferential attachment etc.)
      be sure to do feature extraction after dataset edgelist split to train/val/test set
      Since node features are dependent on edgelist information.
    PS:This class can be reused for graphs other than Twitter as required later
  """
  def __init__(self,av):
    """
      :param av      : args
    """
    PermGnnGraph.__init__(self)
    self.av           = av
    self.dataset_name = av.DATASET_NAME
    self.cites_path   = self.av.DIR_PATH+"/data/Datasets/"+ self.dataset_name + ".txt"
    self.set_adjacency_list()
    self.set_node_features()
    self.set_query_node_info()
    self.max_node_outdegree = max([len(self.adjacency_list[x]) for x in range(self.num_nodes)]) - 1
    assert len(self.adjacency_list) == self.node_features.shape[0]

  def set_node_features(self):
    if self.av.DATASET_NAME in ['AB5K','AB10K']:
      self.node_features = np.identity(self.get_num_nodes())
    else:
      self.node_features = np.random.rand(self.get_num_nodes(),100)

  def set_adjacency_list(self):
    gr = nx.convert_node_labels_to_integers(nx.read_edgelist(self.cites_path))
    #node itself is considered in neighbourhood set
    gr.add_edges_from(zip(nx.nodes(gr),nx.nodes(gr)))
    dict_of_lists = nx.to_dict_of_lists(gr)
    self.adjacency_list = {x: set(dict_of_lists[x]) for x in list(nx.nodes(gr))}
    self.num_nodes = gr.number_of_nodes()

  def set_query_node_info(self):
    self.query_node_list    = list(range(self.get_num_nodes()))
    #list of neighbour sets are cached for lookup in computeLoss (This containes the node itself)
    self.query_node_nbr     = {x: self.adjacency_list[x] for x in self.query_node_list}
    #list of non-neighbour sets are cached for lookup in computeLoss
    self.query_node_non_nbr = {x: set(self.query_node_list) - self.adjacency_list[x] for x in self.query_node_list}
    #removing the node itself from the neighbourhood set
    self.query_node_nbr     = {x:self.query_node_nbr[x]-{x} for x in self.query_node_list}

  def update_query_node_info_tr(self,list_val_edges, list_val_non_edges, list_test_edges, list_test_non_edges):
    for n1,n2 in list_test_edges:
       self.query_node_nbr[n1].discard(n2)
       self.query_node_nbr[n2].discard(n1)
    for n1,n2 in list_val_edges:
       self.query_node_nbr[n1].discard(n2)
       self.query_node_nbr[n2].discard(n1) 
    for n1,n2 in list_test_non_edges:
       self.query_node_non_nbr[n1].discard(n2)
       self.query_node_non_nbr[n2].discard(n1)
    for n1,n2 in list_val_non_edges:
       self.query_node_non_nbr[n1].discard(n2)
       self.query_node_non_nbr[n2].discard(n1) 
