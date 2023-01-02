import numpy as np
import sys
from jax import jit, vmap, value_and_grad
from jax import lax, random, numpy as jnp
from flax import optim
from jax.config import config
config.enable_omnistaging() # Linen requires enabling omnistaging
from util import *

class PNL(nn.Module):
  features: Sequence[int]
  def setup(self):
    # we automatically know what to do with lists, dicts of submodules
    self.layers = [nn.Dense(1),nn.Dense(features = 1, use_bias=False)]
    # for single submodules, we would just write:
    # self.layer1 = nn.Dense(self, feat1)

  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate(self.layers):
      if i ==0: 
        x = lyr(x)
        x = jnp.tanh(x) # nn.sigmoid(x) jnp.tanh(x)
      else:
        x = lyr(x) + x
    return x

# A global variable fmodel
configfname = sys.argv[1]

paramfilename = configfname
params_dic = load_parameters(paramfilename + '.json')
fmodel = nn.Dense(features=1)
fpnl = PNL(features=[1])  # define a neural network

#*********** Start ***********
# Define loss function for the problem 
#*********** Start ***********
@jit
def pred_NN(params, X):
	return fmodel.apply(params, X.reshape(-1,1)).reshape(-1)

@jit
def pred_fpnl(params, Y):
	return fpnl.apply(params, Y.reshape(-1,1)).reshape(-1)

@jit
def otSort_NN(X,Y,param1, param2):
	"""
	Sort effect variable based on the y - g(x)
	Return: the sorted y and x
	"""
	Y_pred = pred_NN(param1, X)
	Y_pnl = pred_fpnl(param2, Y)

	Y_diff = Y_pnl-Y_pred 
	
	ind_sort = jnp.argsort(Y_diff, kind='quicksort')
	Y_diff_sorted = Y_diff[ind_sort]
	X_sorted = X[ind_sort]
	
	y = Y_diff_sorted
	x = X_sorted
	
	return y, x

#******************************************************************
#*********** ANM function parameters Optimizer solver ***********
#******************************************************************
@jit 
def loss_f(params_f,param_pnl,params_d, df_batch, un):
	"""
	loss function for vmap
	"""
	x,y = df_batch[:,0],df_batch[:,1]
	y,x = otSort_NN(X=x,Y=y,param1=params_f,param2=param_pnl)
	noise = jnp.sort(params_d*un)
	vec = y - noise
	return jnp.var(vec) 


## Loss function and gradient of w jit and vmap
val_loss_grad_f = value_and_grad(loss_f)
val_loss_grad_f  = jit(val_loss_grad_f)

## vmap version: when the batches have the same size, it can be vmapped ...
vmap_loss_grad_f_inner = vmap(val_loss_grad_f, in_axes=(None, None, None, None, 1),out_axes=0)  
vmap_loss_grad_f_outer = vmap(vmap_loss_grad_f_inner, in_axes=(None,None, None, 0, 2),out_axes=0)  
vmap_loss_grad_f_outer = jit(vmap_loss_grad_f_outer)
#*********** End ***********

#******************************************************************
# *********** PNL function parameter Optimizer solver  ***********
#******************************************************************
@jit 
def loss_pnl(param_pnl,params_f,params_d, df_batch, un):
	"""
	loss function for vmap
	"""
	x,y = df_batch[:,0],df_batch[:,1]
	y,x = otSort_NN(X=x,Y=y,param1=params_f,param2=param_pnl)
	noise = jnp.sort(params_d*un)
	vec = y - noise
	return jnp.var(vec) 


## Loss function and gradient of w jit and vmap
val_loss_grad_pnl = value_and_grad(loss_pnl)
val_loss_grad_pnl  = jit(val_loss_grad_pnl)

## vmap version: when the batches have the same size, it can be vmapped ...
vmap_loss_grad_pnl_inner = vmap(val_loss_grad_pnl, in_axes=(None, None, None, None, 1),out_axes=0)  
vmap_loss_grad_pnl_outer = vmap(vmap_loss_grad_pnl_inner, in_axes=(None,None, None, 0, 2),out_axes=0)  
vmap_loss_grad_pnl_outer = jit(vmap_loss_grad_pnl_outer)
# *********** END ***********

#******************************************************************
# *********** Distribution parameter Optimizer solver  ***********
#******************************************************************
def make_loss_fun(params_f,params_pnl, df_batch, un):
    """
    loss function for vmap
    """
    def loss_func(thetaH):
        x,y = df_batch[:,0],df_batch[:,1]

        batch_sz,_ = df_batch.shape

        y,x = otSort_NN(X=x,Y=y,param1=params_f,param2=params_pnl)

        noise = jnp.sort(thetaH * un)

        vec = y - noise
        return jnp.var(vec)
    return jit(loss_func)

@jit
def update_d(params_f,params_pnl, params_d, df_batch, un):
    f_opt = make_loss_fun(params_f,params_pnl, df_batch, un)
    fg = grad(f_opt)
    return lax.custom_root(fg, params_d, binary_search, tangent_solve=scalar_solve)
        
vmap_update_d_inner = vmap(update_d, in_axes=(None,None,None,None,1),out_axes=0)  
vmap_update_d_inner = jit(vmap_update_d_inner)
vmap_update_d_outer = vmap(vmap_update_d_inner, in_axes=(None,None,None,0,2),out_axes=0)  
vmap_update_d_outer=jit(vmap_update_d_outer)
# *********** END ***********

#******************************************************************
# ***********Test loss for detecting causal direction***********
#******************************************************************
@jit 
def causal_loss(params_f, params_pnl, params_d, df_batch, un):
    """
    loss function for vmap
    """
    x,y = df_batch[:,0],df_batch[:,1]

    batch_sz,_ = df_batch.shape

    y,x = otSort_NN(X=x,Y=y,param1=params_f,param2=params_pnl)

    noise = jnp.sort(params_d*un)

    vec = y - noise

    return jnp.var(vec)/params_d**2

## vmap version: when the batches have the same size, it can be vmapped ...
vmap_caual_loss_inner_test = vmap(causal_loss, in_axes=(None,None,None,None,1),out_axes=0)  
vmap_caual_loss_outer_test = vmap(vmap_caual_loss_inner_test, in_axes=(None,None,None,0,2),out_axes=0)  
vmap_caual_loss_outer_test = jit(vmap_caual_loss_outer_test)
# *********** END ***********

def get_model_params(filename,dirc,theta_H,subkey):
	if isfile(filename,dirc):
		params_f,params_pnl, params_d = load_model(filename,dirc)
	else:
		params_f = fmodel.init(subkey,[0])
		params_pnl = fpnl.init(subkey,[0])
		params_d = theta_H
	return params_f,params_pnl,params_d

def get_optimizer(learning_rate,params_f):
	optimizer_def = optim.RMSProp(learning_rate=learning_rate) # Choose the method
	optimizer = optimizer_def.create(params_f) # Create the wrapping optimizer with initial parameters
	return optimizer

def optimizer_update(optimizer,grad,learning_rate_fn):
	step = optimizer.state.step
	lr = learning_rate_fn(step)
	return optimizer.apply_gradient(grad, learning_rate=lr)

def test_model(df, params_f, params_pnl, params_d, test_resolution,config, nrep,dirc, subkey):
	key=subkey
	if len(df[:,0])<100:  # If too few data, it will have not enough data for testing with 0.005 resolution
		test_resolution = 0.1
	batches = batchize(df,  test_resolution, len(df[:,0]),config.std)
	batch_sz,_ = batches[0].shape

	optimizer = get_optimizer(config.learning_rate,params_f)
	optimizer_pnl = get_optimizer(config.learning_rate,params_pnl)
	
	if dirc == 'c':
		lr_min=config.lr_min_c
		lr_max=config.lr_max_c 
		steps_per_cycle=config.steps_per_cycle_c

		pnl_lr_min=config.pnl_lr_min_c
		pnl_lr_max=config.pnl_lr_max_c 
		pnl_steps_per_cycle=config.pnl_steps_per_cycle_c

		epoches = config.epoches_c
	else:
		lr_min=config.lr_min_rv
		lr_max=config.lr_max_rv 
		steps_per_cycle=config.steps_per_cycle_rv

		pnl_lr_min=config.pnl_lr_min_rv 
		pnl_lr_max=config.pnl_lr_max_rv
		pnl_steps_per_cycle=config.pnl_steps_per_cycle_rv
		epoches = config.epoches_rv


	learning_rate_fn= create_triangular_schedule(
		lr_min=lr_min, 
		lr_max=lr_min, 
		steps_per_cycle=steps_per_cycle)

	learning_rate_pnl= create_triangular_schedule(
			lr_min=pnl_lr_min, 
			lr_max=pnl_lr_min, 
			steps_per_cycle=pnl_steps_per_cycle)

	loss_test =[]
	params_d_ls = []
	params_f_ls = []
	params_pnl_ls0 = []
	params_pnl_ls1 = []
	res = []
	for i in range(10):
		for _ in range(10):
			key, subkey = random.split(key)
			un = random.normal(subkey,shape=(batch_sz,config.nrep,len(batches)))
			loss_val, grad = vmap_loss_grad_f_outer(optimizer.target,optimizer_pnl.target, params_d, batches, un)
			ave_loss, ave_grad = ave_loss_grad(loss_val,grad)
			optimizer = optimizer_update(optimizer, ave_grad,learning_rate_fn)
			
		key, subkey = random.split(key)    
		un = random.normal(subkey,shape=(batch_sz,50,len(batches)))  # For few samples cases we use larger one
		params_d = np.mean(vmap_update_d_outer(optimizer.target,optimizer_pnl.target, params_d, batches, un))

		for _ in range(10):
			key, subkey = random.split(key)
			un = random.normal(subkey,shape=(batch_sz,config.nrep,len(batches)))
			loss_val, grad = vmap_loss_grad_pnl_outer(optimizer_pnl.target, optimizer.target, params_d, batches, un)
			ave_loss, ave_grad = ave_loss_grad(loss_val,grad)
			optimizer_pnl = optimizer_update(optimizer_pnl, ave_grad,learning_rate_pnl)
		
		params_f = optimizer.target
		params_pnl = optimizer_pnl.target

		params_f_ls.append(unfreeze(params_f)['params'])
		params_pnl_ls0.append(unfreeze(params_pnl)['params']['layers_0'])
		params_pnl_ls1.append(unfreeze(params_pnl)['params']['layers_1'])
		params_d_ls.append(params_d)

		key, subkey = random.split(key)
		un = random.normal(subkey,shape=(batch_sz, nrep, len(batches)))
		loss_test =vmap_caual_loss_outer_test(optimizer.target,optimizer_pnl.target, params_d, batches, un)
		res.append(np.mean(loss_test))

	return res


def optimize_model(df,filename,dirc,config,key):
	loss_tr = tracker()
	theta_tr = tracker()
	loss_pnl_tr = tracker()
	if dirc == 'c':
		lr_min=config.lr_min_c
		lr_max=config.lr_max_c 
		steps_per_cycle=config.steps_per_cycle_c

		pnl_lr_min=config.pnl_lr_min_c
		pnl_lr_max=config.pnl_lr_max_c 
		pnl_steps_per_cycle=config.pnl_steps_per_cycle_c
		epoches = config.epoches_c
	else:
		lr_min=config.lr_min_rv
		lr_max=config.lr_max_rv 
		steps_per_cycle=config.steps_per_cycle_rv

		pnl_lr_min=config.pnl_lr_min_rv 
		pnl_lr_max=config.pnl_lr_max_rv
		pnl_steps_per_cycle=config.pnl_steps_per_cycle_rv
		epoches = config.epoches_rv

	# Make batches out of the data
	batches = batchize(df,config.resolution,config.npos,config.std)
	batch_sz,_ = batches[0].shape
	
	# Define and initialize a optimizer for the NN
	key, subkey = random.split(key)
	params_f,params_pnl, params_d = get_model_params(filename,dirc,config.theta_H,subkey)
	optimizer = get_optimizer(config.learning_rate,params_f)
	optimizer_pnl = get_optimizer(config.learning_rate,params_pnl)

	# Define learning rate schedule
	learning_rate_fn= create_triangular_schedule(
			lr_min=lr_min, 
			lr_max=lr_max, 
			steps_per_cycle=steps_per_cycle)

	learning_rate_pnl= create_triangular_schedule(
			lr_min=pnl_lr_min, 
			lr_max=pnl_lr_max, 
			steps_per_cycle=pnl_steps_per_cycle)

	for i in range(epoches):
		for _ in range(config.epoches_f):
			key, subkey = random.split(key)
			un = random.normal(subkey,shape=(batch_sz,config.nrep,len(batches)))

			loss_val, grad = vmap_loss_grad_f_outer(optimizer.target, optimizer_pnl.target, params_d, batches, un)
			ave_loss, ave_grad = ave_loss_grad(loss_val,grad)
			optimizer = optimizer_update(optimizer, ave_grad,learning_rate_fn)
			loss_tr.add_value(ave_loss)

		key, subkey = random.split(key)    
		un = random.normal(subkey,shape=(batch_sz,50,len(batches)))  # For few samples cases we use larger one
		params_d = np.mean(vmap_update_d_outer(optimizer.target,optimizer_pnl.target, params_d, batches, un))
		theta_tr.add_value(params_d)

		for _ in range(config.epoches_pnl):
			key, subkey = random.split(key)
			un = random.normal(subkey,shape=(batch_sz,config.nrep,len(batches)))

			loss_val, grad = vmap_loss_grad_pnl_outer(optimizer_pnl.target, optimizer.target, params_d, batches, un)
			ave_loss, ave_grad = ave_loss_grad(loss_val,grad)
			optimizer_pnl = optimizer_update(optimizer_pnl, ave_grad, learning_rate_pnl)
			loss_pnl_tr.add_value(ave_loss)

	return optimizer.target,optimizer_pnl.target,params_d,loss_tr,loss_pnl_tr,theta_tr,key

def main(filename,config):

	# Load the data
	key = random.PRNGKey(config.seed)
	x,y = load_data(filename,key,config.std)
	df_sort_c,df_sort_rv = xy_sorted_c_rv(x,y)

	if 'c' in config.exp_type:
		key, subkey = random.split(key)
		opt_params_f_c,opt_params_pnl_c,opt_params_d_c,loss_c,loss_pnl_c,theta_c,key = optimize_model(df_sort_c,filename,'c',config,key)
		
		key, subkey = random.split(key)
		loss_test_c = test_model(df_sort_c, opt_params_f_c, opt_params_pnl_c, opt_params_d_c, config.test_resolution, config, config.nrep, 'c',subkey)

	if 'rv'	in config.exp_type:
		key, subkey = random.split(key)
		opt_params_f_rv,opt_params_pnl_rv,opt_params_d_rv,loss_rv,loss_pnl_rv,theta_rv,key = optimize_model(df_sort_rv,filename,'rv',config,key)

		key, subkey = random.split(key)
		loss_test_rv = test_model(df_sort_rv, opt_params_f_rv, opt_params_pnl_rv, opt_params_d_rv, config.test_resolution, config,config.nrep,'rv', subkey)

	with open('results/acc'+filename+'.txt','w+') as file_writer:
		file_writer.write(filename + ' '+ str(np.mean(loss_test_c))+ ' '+ str(np.mean(loss_test_rv))+ ' '+ str(np.std(loss_test_c))+ ' '+ str(np.std(loss_test_rv)))


if __name__ == "__main__":
    
    # python3 run_realdata.py params 0010 0020 0011

	configfname = sys.argv[1]
	test_list = sys.argv[2:]
	config = Config(configfname)
	for fname in test_list:	
		print('Run experiment '+ fname + '_'+config.exp_type)
		main(fname, config)
