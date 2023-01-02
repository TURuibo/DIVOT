## import
import os.path

import numpy as np
import json
import os 
import pickle

import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random,lax,tree_map, tree_multimap, tree_leaves, grad, value_and_grad, jit, vmap

import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import optim
from flax import serialization

from typing import Any, Callable, Sequence, Optional

from jax.config import config
config.enable_omnistaging() # Linen requires enabling omnistaging

## ************ Neural Networks ************ 
class BBNN(nn.Module):
  features: Sequence[int]

  def setup(self):
    # we automatically know what to do with lists, dicts of submodules
    self.layers = [nn.Dense(feat) for feat in self.features]
    # for single submodules, we would just write:
    # self.layer1 = nn.Dense(self, feat1)

  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)
        # x = nn.sigmoid(x)
    return x
    
class DisNN(nn.Module):
  features: Sequence[int]

  def setup(self):
    # we automatically know what to do with lists, dicts of submodules
    self.layers = [nn.Dense(feat) for feat in self.features]
    # for single submodules, we would just write:
    # self.layer1 = nn.Dense(self, feat1)

  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.sigmoid(x)
        # x = nn.sigmoid(x)
    return nn.sigmoid(x)

## ************ Configuration manangement ************
class Config:
    """docstring for Params"""
    def __init__(self, configfname):
        configfname =   configfname + '.json'
        params_dic = load_parameters(configfname)
    
        self.resolution = params_dic['resolution']
        self.test_resolution = params_dic['test_resolution']

        self.npos = params_dic['npos']
        self.beta =params_dic['beta']
        self.nrep = params_dic['nrep']
        self.decay = params_dic['decay']
        self.learning_rate = params_dic['learning_rate']
        self.epoches_c = params_dic['epoches_c']
        self.epoches_rv = params_dic['epoches_rv']
        self.epoches_f = params_dic['epoches_f']
        self.epoches_pnl = params_dic['epoches_pnl']

        self.theta_H = params_dic['theta_H']
        self.exp_type = params_dic['exp_type']
        self.std = params_dic['std']
        self.seed = params_dic['seed']
        
        self.lr_min_c = params_dic['lr_min_c']
        self.lr_max_c = params_dic['lr_max_c']
        self.steps_per_cycle_c = params_dic['steps_per_cycle_c']
        self.pnl_lr_min_c = params_dic['pnl_lr_min_c']
        self.pnl_lr_max_c = params_dic['pnl_lr_max_c']
        self.pnl_steps_per_cycle_c = params_dic['pnl_steps_per_cycle_c']

        self.lr_min_rv = params_dic['lr_min_rv']
        self.lr_max_rv = params_dic['lr_max_rv']
        self.steps_per_cycle_rv = params_dic['steps_per_cycle_rv']
        self.pnl_lr_min_rv = params_dic['pnl_lr_min_rv']
        self.pnl_lr_max_rv = params_dic['pnl_lr_max_rv']
        self.pnl_steps_per_cycle_rv = params_dic['pnl_steps_per_cycle_rv']

## ************ Ploting functions ************ 
class tracker:
    def __init__(self):
        self.history = []
    def add_value(self, val):
        self.history.append(val)
    def clear(self):
        self.history = []

## ************Save and load models ************
def save_model(params,filename,dirc):
    # reading the data from the file
    with open('nnweights/'+filename+dirc+'.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(params, file)

def isfile(filename,dirc):
    fname = 'nnweights/'+filename+dirc+'.pkl'
    if os.path.isfile(fname):
        return True
    else:
        return False

def load_model(filename,dirc):
    with open('nnweights/'+filename+dirc+'.pkl', 'rb') as file:
        params = pickle.load(file)
    return params

##************ Data loader ************
def load_data(filename,key,std=3):

    data = np.loadtxt('pairs/pair'+filename+'.txt')
    x= data[:,0]
    y= data[:,1]
    x= normalize(x)
    y= normalize(y)
    x,y = x.reshape(-1),y.reshape(-1)

    rangex = std
    rangey = std
    x = x[y<rangey]
    y = y[y<rangey]
    x = x[y>-rangey]
    y = y[y>-rangey]
    y = y[x<rangex]
    x = x[x<rangex]
    y = y[x>-rangex]
    x = x[x>-rangex]

    ind = np.arange(0,len(x),1)
    key, subkey = random.split(key)
    ind = random.permutation(key, ind)

    if len(x)>500:
        ind = ind[:500]

    x = x[ind]
    y = y[ind]
    return x,y

# reading the data from the file
def load_parameters(paramfilename):
    with open(paramfilename, "r") as read_file:
        params_dic = json.load(read_file)
    return params_dic

def xy_sorted_c_rv(x,y):	
	n = len(x)
	df_c = np.zeros([n,2])
	df_c[:,0],df_c[:,1] = x,y
	df_sort_c = sortBycol(df_c,0)

	df_rv = np.zeros([n,2])
	df_rv[:,0],df_rv[:,1] = y,x
	df_sort_rv = sortBycol(df_rv,0)

	return df_sort_c,df_sort_rv

def sortBycol(npa,col):
    ind_sort = np.argsort(npa[:,col])
    return npa[ind_sort,:]

    
##************ Create batches ************ 
# Remove the side batches 
def get_neighbor_matrix_fixed_num(df_sort, resolution):
    """return a matrix with 0, and 1, 0, not a neighbor, and 1, is a neighbor """
    X = df_sort[:,0]   # 0: the column of 'x'
    n = len(X)
    num_suro_samples = int(n * resolution / 2)
    suro_index = np.arange(-num_suro_samples,num_suro_samples,1)
    loop = np.arange(num_suro_samples,n-num_suro_samples,1)
    ## Each col_i represents a position
    ## Elements represent: 1, a neighbor; 0, not a neighbor
    neighbor_matrix = np.zeros([n,n])  
    
    for j in loop:            
        suro_index_pos_j = suro_index + j
        fill_in_fales = X!=X
        fill_in_fales[suro_index_pos_j] = True
        neighbor_matrix[:,j] = fill_in_fales
        neighbor_matrix[j,j] = False
    return neighbor_matrix  # A matrix with True of fales

def get_batch(id_batch, df_sort, neighbor_matrix,  n):
    """given the id of a batch, select its neighbors as a batch"""
    batch_indx = neighbor_matrix[:, id_batch]==1
    df_batch =  df_sort[batch_indx,:]
    return df_batch

def get_batches(data, neighborM, resolution , npos):
    n, _ = np.shape(data) 
    id_batchLoop = np.arange(0,n,int(n/npos))  #
    batches = []
    for id_batch in id_batchLoop:
        batch = get_batch(id_batch=id_batch, df_sort=data, neighbor_matrix=neighborM, n=n) 
        if(len(batch[:,0])>0):
            batches.append(batch)
    return batches

def batchize(df,resolution,npos,std):
    nghM = get_neighbor_matrix_fixed_num(df, resolution)
    if len(df[:,0])<npos:
        npos = len(df[:,0])
    batches = get_batches(data=df, neighborM=nghM, resolution=resolution, npos=npos)

    # Modified for removing the outliers
    # df = df_rm_outliers(batches,df,std)
    # nghM = get_neighbor_matrix_fixed_num(df, resolution)
    # if len(df[:,0])<npos:
    #     npos = len(df[:,0])
    # batches = get_batches(data=df, neighborM=nghM, resolution=resolution, npos=npos)
    #********* END *********

    batches = jnp.array(batches)
    return batches

def check_range(num_std,df,ave,std):
    return np.abs(df-ave ) > num_std * std

def df_rm_outliers(batches,df,num_std):
    outliers = []
    n_batches = np.shape(batches)[0]
    for i in range(n_batches):
        batch = batches[i]
        b_mean, b_std = np.mean(batch[:,1]),np.std(batch[:,1])
        rm_id = check_range(num_std,batch[:,1], b_mean, b_std)
        num_outlier = np.sum(rm_id)
        b_outlier = batch[rm_id, :]
        for j in range(num_outlier):
            tmp = b_outlier[j, :]
            # if np.array(outliers).any(tmp):
            outliers.append(tmp)

    outliers = np.unique(outliers,axis = 0)

    if len(outliers) == 0:
        return df
    num_outliers,_ =np.shape(outliers)

    
    rm = []
    indx = np.arange(0,len(df[:,0]))
    num_outliers,_ =np.shape(outliers)
    for j in range(num_outliers):
        rm_outlier = np.sum(df == outliers[j], axis=1) == 2
        rm_ind = indx[rm_outlier]
        for i in rm_ind:
            rm.append(i)
    rm = np.unique(rm)
    df = np.delete(df, rm, 0)
    return df


## ************ NN loss ************
@jit
def weight_penalty(params, weight_decay = 0.0001):
    weight_penalty_params = tree_leaves(params)
    weight_l2 = sum([jnp.sum(x ** 2)
                    for x in weight_penalty_params
                    if x.ndim > 1])
    return weight_decay  * weight_l2

def ave_loss_grad(loss_val,grad ):
    ave_loss = mean0(mean1(loss_val))
    sum_batch_ave_grad_thetaH =  mean0(mean1(unfreeze(grad['thetaH'])))
    ave_grad_NN_rep = tree_map(mean1, unfreeze(grad['NN']))
    sum_batch_ave_grad_NN = tree_map(mean0, unfreeze(ave_grad_NN_rep))
    ave_grad ={'thetaH':sum_batch_ave_grad_thetaH,'NN':freeze(sum_batch_ave_grad_NN)}
    return ave_loss,ave_grad

def ave_loss_grad_f(loss_val,grad ):
    ave_loss = mean0(mean1(loss_val))   
    ave_grad_f_rep = tree_map(mean1, unfreeze(grad))
    sum_batch_ave_grad_f = tree_map(mean0, unfreeze(ave_grad_f_rep))
    ave_grad =freeze(sum_batch_ave_grad_f)
    return ave_loss,ave_grad

def ave_loss_grad(loss_val,grad ):
    ave_loss = mean0(mean1(loss_val))   
    ave_grad_rep = tree_map(mean1, unfreeze(grad))
    sum_batch_ave_grad = tree_map(mean0, unfreeze(ave_grad_rep))
    ave_grad =freeze(sum_batch_ave_grad)
    return ave_loss,ave_grad

def ave_params_f(dict_list):
    """Input: the params element extracted from flax neural network parameters """
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = np.mean([d[key] for d in dict_list], axis=0)
    return freeze({'params':mean_dict})

def ave_params_pnl(dict_list):
    """Input: the params element extracted from flax neural network parameters """
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = np.mean([d[key] for d in dict_list], axis=0)
    layers_0 ={'layers_0':mean_dict}
    return freeze({'params':layers_0})

def ave_params_pnl_2layers(dict_list0,dict_list1):
    """Input: the params element extracted from flax neural network parameters """
    mean_dict = {}
    mean_dict1 = {}
    for key in dict_list0[0].keys():
        mean_dict[key] = np.mean([d[key] for d in dict_list0], axis=0)
    for key in dict_list1[0].keys():
        mean_dict1[key] = np.mean([d[key] for d in dict_list1], axis=0)
    layers ={'layers_0':mean_dict,'layers_1':mean_dict1}
    return freeze({'params':layers})
    mean_dict = {}
    mean_dict1 = {}
    for key in dict_list['layers_0'][0].keys():
        mean_dict[key] = np.mean([d[key] for d in dict_list['layers_0']], axis=0)
    for key in dict_list['layers_1'][0].keys():
        mean_dict1[key] = np.mean([d[key] for d in dict_list['layers_1']], axis=0)
    layers ={'layers_0':mean_dict,'layers_1':mean_dict1}
    return freeze({'params':layers})

## ************ Vmap for BbNN ************
def normalize(x):
    return (x-np.mean(x))/np.std(x)
def mean1(x):
    return jnp.mean(x,axis = 1)
def mean0(x):
    return jnp.mean(x,axis = 0)
def sum0(x):
    return jnp.sum(x,axis = 0)

## ************ Learning rate schedule ************
def create_triangular_schedule(lr_min=0.001, lr_max=0.1, steps_per_cycle=200):
  top = (steps_per_cycle + 1) // 2
  def learning_rate_fn(step):
    cycle_step = step % steps_per_cycle
    if cycle_step < top:
      lr = lr_min + cycle_step/top * (lr_max - lr_min)
    else:
      lr = lr_max - ((cycle_step - top)/top) * (lr_max - lr_min)
    return lr
  return learning_rate_fn
  
# ************ Optimizer solver ************
def binary_search(func, x0, low=0.0, high=100.0):
      del x0  # unused

      def cond(state):
        low, high = state
        midpoint = 0.5 * (low + high)
        return (low < midpoint) & (midpoint < high)

      def body(state):
        low, high = state
        midpoint = 0.5 * (low + high)
        update_upper = func(midpoint) > 0
        low = jnp.where(update_upper, low, midpoint)
        high = jnp.where(update_upper, midpoint, high)
        return (low, high)

      solution, _ = lax.while_loop(cond, body, (low, high))
      return solution

def scalar_solve(f, y):
      return y / f(1.0)  



# ************ test scores************
def NormLoss(loss_tracker, theta_tracker):
    theta_ls = []
    num_rep = len(loss_tracker.history)/len(theta_tracker.history)
    for i in range(len(theta_tracker.history)):
        for _ in range(int(num_rep)):
            theta_ls.append(theta_tracker.history[i]) 
    res =[loss_tracker.history[i]/theta_ls[i]**2 for i in range(len(loss_tracker.history))]
    return res

# ************ plot figures************

def save_plots(df, loss_ls,loss_pnl, theta_ls,filename,dirc):
    plt.figure(figsize=(20,4))
    plt.subplot(1,4,1)
    plt.scatter(df[:,0],df[:,1])
    plt.title( f"nsample: { len(df[:,0])} exp"+filename+dirc)

    plt.subplot(1,4,2)
    plt.title('loss')
    x = jnp.arange(1,len(loss_ls.history)+1,1)
    plt.plot(x,np.log10(loss_ls.history))

    plt.subplot(1,4,3)
    plt.title('loss pnl')
    x = jnp.arange(1,len(loss_pnl.history)+1,1)
    plt.plot(x,np.log10(loss_pnl.history))

    plt.subplot(1,4,4)
    plt.title('theta')
    x = jnp.arange(1,len(theta_ls.history)+1,1)
    plt.plot(x,np.log10(theta_ls.history))

    plt.savefig('finetune/'+filename+dirc+'.png')




