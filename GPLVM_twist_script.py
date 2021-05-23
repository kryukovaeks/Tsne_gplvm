
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch.nn import Parameter
import matplotlib
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#from GPy import kern
#from GPy.models import GPLVM
#%matplotlib inline

sizeTitle = 24
sizeAxis = 14
#smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.6.0')
pyro.set_rng_seed(1)




class Twist_gplvm():
    def __init__(self,var,dim):
        self.var = var
        self.dim = dim

    def plot_twist_project(self,data):

        twist=np.load(data)
        data = torch.tensor(twist[:,:3], dtype=torch.get_default_dtype())
        y = data.t()
        capture_time = y.new_tensor(twist[:,3])

        time=capture_time
        # we setup the mean of our prior over X, 1st axis
        X_prior_mean = torch.zeros(y.size(1), self.dim)
        X_prior_mean[:, 0] = time
        kernel = gp.kernels.RBF(input_dim=self.dim, lengthscale=torch.ones(self.dim))

        # we clone here so that we don't change our prior during the course of training
        X = Parameter(X_prior_mean.clone())


        # we will use SparseGPRegression model with num_inducing=2;
        # initial values for Xu are sampled randomly from X_prior_mean
        Xu = stats.resample(X_prior_mean.clone(), self.dim)
        gplvm = gp.models.SparseGPRegression(X, y, kernel, Xu, noise=torch.tensor(0.01), jitter=1e-5)
        # we use `.to_event()` to tell Pyro that the prior distribution for X has no batch_shape
        gplvm.X = pyro.nn.PyroSample(dist.Normal(X_prior_mean, self.var).to_event())
        gplvm.autoguide("X", dist.Normal)

        # note that training is expected to take a minute or so
        losses = gp.util.train(gplvm, num_steps=4000)

        """# let's plot the loss curve after 4000 steps of training
        plt.plot(losses)
        plt.show()"""
        gplvm.mode = "guide"
        X = gplvm.X  # draw a sample from the guide of the variable X
        X = gplvm.X_loc.detach().numpy()


        fig = plt.figure(figsize = plt.figaspect(1))

        matplotlib.rcParams.update({'font.size':sizeAxis})

        if self.dim==2:
            plt.scatter(X[:,0], X[:,1], c=twist[:,3])
        if self.dim==1:
            plt.scatter(X, [1]*len(X), c=twist[:, 3])





        plt.title("GPLVM on cylinder with a slit:" +"\n"+ "prior variance = %s" %(self.var)+ ", dim=%s" %(self.dim), fontsize=16)
        #plt.title('TSNE: perp = %d' %(perp)+', iter = %d' %(iter), fontsize=sizeTitle)

        plt.show()
