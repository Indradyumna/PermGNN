from common import logger
import os
import torch

def load_model(av):
  """
    :param av           : args
    :return checkpoint  : dict containing optimizer and model state dicts and last run epoch no. 
  """
  load_dir = os.path.join(av.DIR_PATH, "savedModels")
  if not os.path.isdir(load_dir):
    raise Exception('{} does not exist'.format(load_dir))
  name = av.DATASET_NAME
  if av.TASK !="":
    name = av.TASK + "_" + name
  name = name + "_tfrac_" + str(av.TEST_FRAC) + "_vfrac_" + str(av.VAL_FRAC)
  load_path = os.path.join(load_dir, name)
  logger.info("loading model from %s",load_path)
  checkpoint = torch.load(load_path)
  return checkpoint

def save_model(av,model,optimizerPerm, optimizerFunc, epoch, saveAllEpochs = True):
  """
    :param av            : args
    :param model         : nn model whose state_dict is to be saved
    :param optimizerPerm : state_dict is to be saved
    :param optimizerFunc : state_dict is to be saved
    :param epoch         : epoch no. 
    :param saveAllEpochs : if True, dumps model weights of all intermediate epochs
                           Sometimes set to False due to memory considerations
    :return              : None
  """
  save_dir = os.path.join(av.DIR_PATH, "savedModels")
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  name = av.DATASET_NAME
  if av.TASK !="":
    name = av.TASK + "_" + name
  name = name + "_tfrac_" + str(av.TEST_FRAC) + "_vfrac_" + str(av.VAL_FRAC)
  save_prefix = os.path.join(save_dir, name)

  if saveAllEpochs:
    save_path = '{}_epoch_{}'.format(save_prefix, epoch)
    logger.info("saving model to %s",save_path)
    #Saving per epoch model state - Needed to obtain embeddings wrt specific epoch
    output = open(save_path, mode="wb")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_perm_state_dict': optimizerPerm.state_dict(),
            'optimizer_func_state_dict': optimizerFunc.state_dict(),
            'av' : av,
            }, output)
    output.close() 

  #Also storing the latest epoch (will be overwritten at each epoch) 
  #For easier restore of latest run epoch
  logger.info("saving model to %s",save_prefix)
  output = open(save_prefix, mode="wb")
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_perm_state_dict': optimizerPerm.state_dict(),
            'optimizer_func_state_dict': optimizerFunc.state_dict(),
            'av' : av,
            }, output)
  output.close()

def init_optimizers(av,model):
  """
    :param av             : args 
    :param model          : nn model
    :return optimizerPerm : Maximixing loss wrt permutation network parameters
    :return optimizerFunc : Minimizing loss wrt LSTM+FC network parameters
  """
  permutation_params = ["permNet.linear1.weight", "permNet.linear1.bias", "permNet.linear2.weight", "permNet.linear2.bias"]
  lstm_params = ["lstm.weight_ih_l0" , "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0", "fully_connected_layer.weight", "fully_connected_layer.bias"]
  if av.OPTIM == "SGD":
    optimizerPerm = torch.optim.SGD([p for n, p in model.named_parameters() if n in permutation_params], lr=av.LEARNING_RATE_PERM)
    optimizerFunc = torch.optim.SGD([p for n, p in model.named_parameters() if n in lstm_params], lr=av.LEARNING_RATE_FUNC)
  elif av.OPTIM == "Adam":
    optimizerPerm = torch.optim.Adam([p for n, p in model.named_parameters() if n in permutation_params], lr=av.LEARNING_RATE_PERM)
    optimizerFunc = torch.optim.Adam([p for n, p in model.named_parameters() if n in lstm_params], lr=av.LEARNING_RATE_FUNC)
  else:
    raise NotImplementedError()  
  return optimizerPerm,optimizerFunc

def set_learnable_parameters(model,isMaxPhase): 
  """
    :param model      : nn model
    :param isMaxPhase : If True, freeze LSTM+FC parameters and unfreeze Permutation parameters
                        If False, unfreeze LSTM+FC parameters and freeze Permutation parameters
    :return           : None
  """  
  permutation_params = ["permNet.linear1.weight", "permNet.linear1.bias", "permNet.linear2.weight", "permNet.linear2.bias"]
  lstm_params = ["lstm.weight_ih_l0" , "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0", "fully_connected_layer.weight", "fully_connected_layer.bias"]
  for name, param in model.named_parameters():
    if name in lstm_params:
      param.requires_grad = not isMaxPhase
    if name in permutation_params:
      param.requires_grad = isMaxPhase

def cudavar(av, x):
    """Adapt to CUDA or CUDA-less runs.  Annoying av arg may become
    useful for multi-GPU settings."""
    return x.cuda() if av.has_cuda and av.want_cuda else x

def pytorch_sample_gumbel(av,shape, eps=1e-20):
  #Sample from Gumbel(0, 1)
  U = cudavar(av,torch.rand(shape).float())
  return -torch.log(eps - torch.log(U + eps))
