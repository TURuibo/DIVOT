from util import *
from jax import grad, value_and_grad, jit,vmap, random
import sys

@jit
def loss_jax(params, df_batch,un):
    x,y = df_batch[:,0],df_batch[:,1]
    vec = jnp.sort(y-params['w']*x) - jnp.sort(params['theta']*un)
    return jnp.var(vec)

val_and_grad = value_and_grad(loss_jax)
vmap_val_and_grad_inner = vmap(val_and_grad, in_axes=(None,None,1),out_axes=0)  
vmap_val_and_grad_outer = vmap(vmap_val_and_grad_inner, in_axes=(None,0,2),out_axes=0)  
vmap_val_and_grad_outer = jit(vmap_val_and_grad_outer)


def batch_test(df,resolution,npos):
    nghM = get_neighbor_matrix_fixed_num(df, resolution)
    batches = get_batches(data=df, neighborM=nghM, resolution=resolution, npos=npos)
    batches = jnp.array(batches)
    df_batch = batches[0]
    batch_sz,_ = df_batch.shape
    return batches,batch_sz


def test(batches, key_seed=42, step_sz_w = 1.0,step_sz_theta = 1.0, exp = 200, nrep = 100):
    key = random.PRNGKey(key_seed)
    df_batch = batches[0]
    batch_sz,_ = df_batch.shape
    
    theta =0.2
    w = 0.1
    params = {'w': w, 'theta':theta}

    loss_res = []
    params_res = []
    w_res = []
    t_res = []
    gradt_res = []
    gradw_res = []

    for j in range(exp):
        key, subkey = random.split(key)
        un = random.uniform(subkey,shape=(batch_sz,nrep,len(batches)),minval=0.0, maxval=1.0)
        loss_val,grad = vmap_val_and_grad_outer(params, batches, un)

        ave_loss = np.mean(loss_val)
        ave_grad = tree_map(np.mean, grad)

        params['w'] -= step_sz_w * ave_grad['w']
        params['theta'] -= step_sz_theta * ave_grad['theta']

        loss_res.append(ave_loss)
        w_res.append(params['w'])
        t_res.append(params['theta'])
        gradw_res.append(ave_grad['w'])
        gradt_res.append(ave_grad['theta'])
 
    return loss_res,w_res,t_res,gradw_res,gradt_res

def get_loss(loss_res,params):
	return np.mean(loss_res[-10:])/params

def main():
	res = 0
	# Get the parameters of the experiment
	num_exp,nsamples,resolution,npos,nexp,sz_w,sz_t= int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7])
	theta_est = []
	for seed in range(num_exp):
		sys.stdout.write("\r Experiment: %i " % seed)

		def f_t(x):
		    return x

		key = random.PRNGKey(seed)
		key, subkey = random.split(key)

		# Generate samples with additional noise
		ksample, knoise = random.split(subkey)
		x_samples = random.uniform(subkey,shape=(nsamples, 1),minval=-1, maxval=1)
		y_samples = np.array([f_t(x) for x in x_samples])
		y_samples += 1.0*random.uniform(knoise,shape=(nsamples, 1),minval=0.0, maxval=1.0)
		
		x= x_samples.reshape(-1)
		y= y_samples.reshape(-1)
		n = nsamples

		df_c = np.zeros([n,2])
		df_c[:,0],df_c[:,1] = x,y
		df_sort_c = sortBycol(df_c,0)

		df_rv = np.zeros([n,2])
		df_rv[:,0],df_rv[:,1] = y,x
		df_sort_rv = sortBycol(df_rv,0)

		# Test in the causal direction
		c_batches,c_batch_sz = batch_test(df_sort_c,resolution,npos)
		c_loss_res, c_w_res,c_t_res,c_gradw_res,c_gradt_res= test(c_batches, key_seed = 42, step_sz_w = sz_w,step_sz_theta = sz_t, exp = nexp,nrep = 50)
		loss_c = np.mean(c_loss_res[-10:])/np.mean(c_t_res[-10:])**2

		# Test in the reverse direction
		rv_batches,rv_batch_sz = batch_test(df_sort_rv,resolution,npos)
		rv_loss_res, rv_w_res,rv_t_res,rv_gradw_res,rv_gradt_res= test(rv_batches,key_seed = 42, step_sz_w = sz_w,step_sz_theta=sz_t, exp = nexp,nrep = 50)
		loss_rv = np.mean(rv_loss_res[-10:])/np.mean(rv_t_res[-10:])**2

		theta_est.append(np.mean(c_t_res[-10:]))

		if loss_c < loss_rv:
			res += 1

	print('\nnsamples:',nsamples,"resolution:",resolution,"npos:",npos, 'acc:', res/num_exp, 'theta:', np.mean(theta_est),'std theta:', np.std(theta_est))

if __name__ == '__main__':
	main()










