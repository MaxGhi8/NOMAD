
from jax.flatten_util import ravel_pytree
from jax.example_libraries.stax import Dense, Gelu
from jax.example_libraries import stax, optimizers
import os
import timeit

import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import norm
from jax import random, grad, jit, vmap 
from jax.experimental.ode import odeint
from functools import partial 
from torch.utils import data
from tqdm import trange
import itertools
import argparse

# def get_freer_gpu():
#     os.system('nvidia-smi -q -d Memory |grep Free >tmp')
#     memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
#     return str(np.argmax(memory_available))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="False"

class DataGenerator(data.Dataset):
    def __init__(self, u, y, s,
                 batch_size=100, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs,outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx,:,:]
        u  = self.u[idx,:,:]
        y = self.y[idx,:,:]
        inputs = (u, y)
        return inputs, s

# Geneate training data corresponding to one input sample
def generate_one_datum(freq, m=500, P=500):
    X = jnp.linspace(0,1, num=m)
    u = jnp.cos(2*jnp.pi*freq*X)
    u_fn = lambda x, t: jnp.interp(t, X.flatten(), u)
    u = vmap(u_fn, in_axes=(None,0))(0.0, X)
    y_train = jnp.linspace(0, 1, P)
    s_train = odeint(u_fn, 0.0, y_train)
    return u, y_train, s_train

# Geneate training data corresponding to N input sample
def generate_data(freqs, N, m):
  gen_fn = jit(lambda freq: generate_one_datum(freq, m))
  u_train, y_train, s_train = vmap(gen_fn)(freqs)
  return u_train, y_train, s_train

class operator_model:
    def __init__(self,branch_layers, trunk_layers, NN_layers = None, m=100, P=100,n=None, decoder=None, ds=None):

        # Branch net
        seed = np.random.randint(low=0, high=100000)
        self.branch_init, self.branch_apply = self.init_NN(branch_layers, activation=Gelu)
        self.in_shape = (-1, branch_layers[0])
        self.out_shape, branch_params = self.branch_init(random.PRNGKey(seed), self.in_shape)

        if decoder=="nonlinear":
            self.fwd = self.NOMAD
            # Trunk net
            seed = np.random.randint(low=0, high=100000)
            self.trunk_init, self.trunk_apply = self.init_NN(trunk_layers, activation=Gelu)
            self.in_shape = (-1, trunk_layers[0])
            self.out_shape, trunk_params = self.trunk_init(random.PRNGKey(seed), self.in_shape)

            params = (trunk_params, branch_params)

        if decoder=="linear":
            self.fwd = self.DeepONet
            # Trunk net
            seed = np.random.randint(low=0, high=100000)
            self.trunk_init, self.trunk_apply = self.init_NN_final_act(trunk_layers, activation=Gelu)
            self.in_shape = (-1, trunk_layers[0])
            self.out_shape, trunk_params = self.trunk_init(random.PRNGKey(seed), self.in_shape)

            params = (trunk_params, branch_params)

        if decoder=="nonlinear_v2":
            self.fwd = self.NOMAD_v2

            # Trunk net
            seed = np.random.randint(low=0, high=100000)
            self.trunk_init, self.trunk_apply = self.init_NN_final_act(trunk_layers, activation=Gelu)
            self.in_shape = (-1, trunk_layers[0])
            self.out_shape, trunk_params = self.trunk_init(random.PRNGKey(seed), self.in_shape)
            
            # final NN
            seed = np.random.randint(low=0, high=100000)
            self.NN_init, self.NN_apply = self.init_NN(NN_layers, activation=Gelu)
            self.in_shape = (-1, NN_layers[0])
            self.out_shape, NN_params = self.NN_init(random.PRNGKey(seed), self.in_shape)

            params = (trunk_params, branch_params, NN_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init,self.opt_update,self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=100, 
                                                                      decay_rate=0.99))
        self.opt_state = self.opt_init(params)
        # Logger
        self.itercount = itertools.count()
        self.loss_log = []


        self.n  = n
        self.ds = ds
        self.m = m
        self.P = P

    def init_NN(self, Q, activation=Gelu):
        layers = []
        num_layers = len(Q)
        if num_layers < 2:
            net_init, net_apply = stax.serial()
        else:
            for i in range(0, num_layers-2):
                layers.append(Dense(Q[i+1]))
                layers.append(activation)
            layers.append(Dense(Q[-1]))
            net_init, net_apply = stax.serial(*layers)
        return net_init, net_apply

    def init_NN_final_act(self, Q, activation=Gelu):
        layers = []
        num_layers = len(Q)
        if num_layers < 2:
            net_init, net_apply = stax.serial()
        else:
            for i in range(0, num_layers-2):
                layers.append(Dense(Q[i+1]))
                layers.append(activation)
            layers.append(Dense(Q[-1]))
            layers.append(stax.Tanh)
            net_init, net_apply = stax.serial(*layers)
        return net_init, net_apply

    @partial(jax.jit, static_argnums=0)
    def NOMAD(self, params, inputs):
        trunk_params, branch_params = params
        inputsu, inputsy = inputs
        # print('inputsu',inputsu.shape) # (100, 500, 1)
        # print('inputsy',inputsy.shape) # (100, 500, 1)
        # print('inputsu reshaped', inputsu.reshape(inputsu.shape[0], 1, inputsu.shape[1]).shape) # (100, 1, 500)
        b = self.branch_apply(branch_params, inputsu.reshape(inputsu.shape[0], 1, inputsu.shape[1])) 
        # print('b',b.shape) # (100, 1, latent_dim)
        b = jnp.tile(b, (1,inputsy.shape[1],1))
        # print(b[:, 0, :] - b[:, 42, :]) # all zeros
        # print('reshaped b',b.shape) # (100, 500, latent_dim)
        # tmp = jnp.tile(inputsy,(1,1,b.shape[-1]//inputsy.shape[-1]))
        # print('reshaped inputsy',tmp.shape) # (100, 500, latent_dim)
        # print(tmp[0, : , 0] - tmp[42, :, 42]) # all zeros
        inputs_recon = jnp.concatenate((jnp.tile(inputsy,(1,1,b.shape[-1]//inputsy.shape[-1])), b), axis=-1)
        # print('inputs_recon',inputs_recon.shape) # (100, 500, 2*latent_dim)
        out = self.trunk_apply(trunk_params, inputs_recon)
        # print('out',out.shape) # (100, 500, 1)
        return out

    @partial(jax.jit, static_argnums=0)
    def NOMAD_v2(self, params, inputs):
        trunk_params, branch_params, NN_params = params
        inputsxu, inputsy = inputs # (100, 500, 1), (100, 500, 1)
        
        # Branch net
        inputsxu = inputsxu.reshape(inputsxu.shape[0], 1, inputsxu.shape[1]*inputsxu.shape[2]) # (100, 1, 500)
        b = self.branch_apply(branch_params, inputsxu) # (100, 1, latent_dim)
        b = jnp.tile(b, (1, self.m, 1)) # (100, 500, latent_dim)

        # Trunk net
        t = self.trunk_apply(trunk_params, inputsy) # (100, 500, latent_dim)
        
        # non-linear decoder
        NN_input = jnp.concatenate((t, b), axis=-1) # (100, 500, 2*latent_dim)
        out = self.NN_apply(NN_params, NN_input) 

        return out

    @partial(jax.jit, static_argnums=0)
    def DeepONet(self, params, inputs):
        trunk_params, branch_params = params
        inputsxu, inputsy = inputs
        # print('inputsxu',inputsxu.shape) # (100, 500, 1)
        # print('inputsy',inputsy.shape) # (100, 500, 1)
        # print('inputsxu reshaped', inputsxu.reshape(inputsxu.shape[0],1,inputsxu.shape[1]*inputsxu.shape[2]).shape) # (100, 1, 500)
        t = self.trunk_apply(trunk_params, inputsy).reshape(inputsy.shape[0], inputsy.shape[1], self.ds, self.n)
        # print('t', self.trunk_apply(trunk_params, inputsy).shape) # (100, 500, latent_dim)
        # print('reshaped t',t.shape) # (100, 500, 1, latent_dim)
        b = self.branch_apply(branch_params, inputsxu.reshape(inputsxu.shape[0],1,inputsxu.shape[1]*inputsxu.shape[2]))
        # print('b',b.shape) # (100, 1, latent_dim)
        b = b.reshape(b.shape[0],int(b.shape[2]/self.ds),self.ds)
        # print('reshaped b',b.shape) # (100, latent_dim, 1)
        Guy = jnp.einsum("ijkl,ilk->ijk", t,b)
        # print('Guy',Guy.shape) # (100, 500, 1)
        return Guy
        
    @partial(jax.jit, static_argnums=0)
    def loss(self, params, batch):
        inputs, y = batch
        y_pred = self.fwd(params,inputs)
        loss = np.mean((y.flatten() - y_pred.flatten())**2)
        return loss    

    @partial(jax.jit, static_argnums=0)
    def L2error(self, params, batch):
        inputs, y = batch
        y_pred = self.fwd(params,inputs)
        return norm(y.flatten() - y_pred.flatten(), 2)/norm(y.flatten(),2)

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch)
        return self.opt_update(i, g, opt_state)

    def train(self, train_dataset, test_dataset, nIter = 10000):
        train_data = iter(train_dataset)
        test_data  = iter(test_dataset)

        pbar = trange(nIter)
        for it in pbar:
            train_batch = next(train_data)
            test_batch  = next(test_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, train_batch)
            
            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                loss_train = self.loss(params, train_batch)
                loss_test  = self.loss(params, test_batch)

                errorTrain = self.L2error(params, train_batch)
                errorTest  = self.L2error(params, test_batch)

                self.loss_log.append(loss_train)

                pbar.set_postfix({'Training loss': loss_train, 
                                  'Testing loss' : loss_test,
                                  'Test error':    errorTest,
                                  'Train error':   errorTrain})

    @partial(jit, static_argnums=(0,))
    def predict(self, params, inputs):
        s_pred = self.fwd(params,inputs)
        return s_pred

    def count_params(self):
        params = self.get_params(self.opt_state)
        params_flat, _ = ravel_pytree(params)
        print("The number of model parameters is:",params_flat.shape[0])

def main(n, decoder):
    TRAINING_ITERATIONS = 20000
    P = 500
    m = 500
    num_train = 1000
    num_test  = 1000
    training_batch_size = 100
    du = 1
    dy = 1
    ds = 1
    Nx = 100

    key_train = random.PRNGKey(0)
    _, subkey = random.split(key_train)
    train_freqs = random.uniform(subkey, minval=0, maxval=10, shape=(num_train,))
    U_train, y_train, s_train = generate_data(train_freqs, num_train, m)
    key_test = random.PRNGKey(12345)
    _, subkey = random.split(key_test)
    test_freqs = random.uniform(subkey, minval=0, maxval=10, shape=(num_test,))
    U_test, y_test, s_test = generate_data(test_freqs, num_test, m)

    U_train = jnp.asarray(U_train)
    y_train = jnp.asarray(y_train)
    s_train = jnp.asarray(s_train)

    U_test = jnp.asarray(U_test)
    y_test = jnp.asarray(y_test)
    s_test = jnp.asarray(s_test)

    U_train = jnp.reshape(U_train,(num_train,m,du))
    y_train = jnp.reshape(y_train,(num_train,P,dy))
    s_train = jnp.reshape(s_train,(num_train,P,ds))

    U_test = jnp.reshape(U_test,(num_test,m,du))
    y_test = jnp.reshape(y_test,(num_test,P,dy))
    s_test = jnp.reshape(s_test,(num_test,P,ds))

    train_dataset = DataGenerator(U_train, y_train, s_train, training_batch_size)
    train_dataset = iter(train_dataset)
    test_dataset = DataGenerator(U_test, y_test, s_test, training_batch_size)
    test_dataset = iter(test_dataset)

    if decoder=="nonlinear":
        branch_layers = [m,   100, 100, 100, 100, 100, ds*n]
        trunk_layers  = [ds*n*2, 100, 100, 100, 100, 100, ds]
        NN_layers = None
    if decoder=="nonlinear_v2":
        branch_layers = [m,   100, 100, 100, 100, 100, ds*n]
        trunk_layers  = [dy,  100, 100, 100, 100, 100, ds*n]
        NN_layers = [2*ds*n, 100, 100, 100, 100, 100, 1]
    elif decoder=="linear":
        branch_layers = [m,   100, 100, 100, 100, 100, ds*n]
        trunk_layers  = [dy,  100, 100, 100, 100, 100, ds*n]
        NN_layers = None

    model = operator_model(branch_layers, trunk_layers, NN_layers=NN_layers, m=m, P=P,n=n, decoder=decoder, ds=ds)
    model.count_params()

    start_time = timeit.default_timer()
    model.train(train_dataset, test_dataset, nIter=TRAINING_ITERATIONS)
    elapsed = timeit.default_timer() - start_time
    print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)

    del train_dataset, test_dataset

    params = model.get_params(model.opt_state)

    s_pred_test = np.zeros_like(s_test)
    for i in range(0,num_test,100):
        idx = i + np.arange(0,100)
        s_pred_test[idx] = model.predict(params, (U_test[idx], y_test[idx]))
    test_error_u = []
    for i in range(0,num_train):
        test_error_u.append(norm(s_test[i,:,0]- s_pred_test[i,:,0],2)/norm(s_test[i,:,0],2))
    print("The average test u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))

    s_pred_train = np.zeros_like(s_train)
    for i in range(0,num_train,100):
        idx = i + np.arange(0,100)
        s_pred_train[idx] = model.predict(params, (U_train[idx], y_train[idx]))
    train_error_u = []
    for i in range(0,num_test):
        train_error_u.append(norm(s_train[i,:,0]- s_pred_train[i,:,0],2)/norm(s_train[i,:,0],2))
    print("The average train u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(train_error_u),np.std(train_error_u),np.min(train_error_u),np.max(train_error_u)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('n', metavar='n', type=int, nargs='+', help='Latent dimension of the solution manifold')
    parser.add_argument('decoder', metavar='decoder', type=str, nargs='+', help='Type of decoder. Choices a)"linear" b)"nonlinear"')

    args = parser.parse_args()
    n = args.n[0]
    decoder = args.decoder[0]
    main(n,decoder)
