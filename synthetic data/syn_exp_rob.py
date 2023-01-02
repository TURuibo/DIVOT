from util import *
from jax import grad, value_and_grad, jit,vmap, random
import sys

@jit
def loss_jax(params, df_batch,un):
    x,y = df_batch[:,0],df_batch[:,1]
    vec = jnp.sort(y) - jnp.sort(params*un)
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


def test(batches,key,step_sz = 1.0,exp = 200,nrep = 100):
    df_batch = batches[0]
    batch_sz,_ = df_batch.shape
    theta_H =0.2
    params = theta_H

    loss_res = []
    t_res = []
    gradt_res = []

    for j in range(exp):
        key, subkey = random.split(key)
        # un = random.uniform(subkey,shape=(batch_sz,nrep,len(batches)),minval=0.0, maxval=1.0)
        un = random.beta(subkey,shape=(batch_sz,nrep,len(batches)),a=0.5,b=0.5)
        # un = random.normal(subkey,shape=(batch_sz,nrep,len(batches)))

        loss_val,grad = vmap_val_and_grad_outer(params, batches, un)
        ave_loss,ave_grad = np.mean(loss_val),np.mean(grad)
        params -= step_sz * ave_grad
        loss_res.append(ave_loss)
        t_res.append(params)
        gradt_res.append(ave_grad)
        # if j%20==0:
        #     sys.stdout.write("\rEpoch: %i" % j)  
    return loss_res,t_res,gradt_res,params

def get_loss(loss_res,params):
	return np.mean(loss_res[-10:])/params**2

def main():
	res = 0
	num_exp,nsamples,resolution,npos= int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4])
	theta_est = []
	for seed in range(num_exp):
		sys.stdout.write("\r Experiment: %i " % seed)

		def f_t(x):
		    # return 0.1*((2.5*x)**3 - x)
		    return x
		    # return jnp.sin(4*x)
		    # if x < 0:
		    #     return 0.5*x**3 - x
		    # else:   
		    #     return 1 - 0.5*x**3 + x

		key = random.PRNGKey(seed)
		key, subkey = random.split(key)

		# Generate samples with additional noise
		ksample, knoise = random.split(subkey)
		x_samples = random.uniform(subkey,shape=(nsamples, 1),minval=-1, maxval=1)

		# y_samples = vmap(f_t)(x_samples)
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

		c_batches,c_batch_sz = batch_test(df_sort_c,resolution,npos)
		c_loss_res, c_t_res, c_gradt_res, params_c = test(c_batches, key, step_sz = 1.0,exp = 100,nrep = 50)
		dir_c_loss = get_loss(c_loss_res,params_c)
		theta_est.append(params_c)

		rv_batches,rv_batch_sz = batch_test(df_sort_rv,resolution,npos)
		rv_loss_res,rv_t_res,rv_gradt_res,params_rv = test(rv_batches, key, step_sz = 1.0,exp = 100,nrep = 50)
		dir_rv_loss= get_loss(rv_loss_res,params_rv)
		if dir_c_loss < dir_rv_loss:
			res += 1

	print('\nnsamples:',nsamples,"resolution:",resolution,"npos:",npos, 'acc:', res/num_exp, 'theta:', np.mean(theta_est),'std theta:', np.std(theta_est))

if __name__ == '__main__':
	main()










