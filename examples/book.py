import particles
from particles import state_space_models as ssm
from particles import distributions as dists

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from numpy import random


# #########
# Chapter 4

class StochVol(ssm.StateSpaceModel):
    default_params = {'mu': -1., 'rho':0.9, 'sigma':1.}
    def PX0(self):
        sig0 = self.sigma / np.sqrt(1. - self.rho**2)
        return dists.Normal(loc=self.mu, scale=sig0)
    def PX(self, t, xp):
        return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), scale=self.sigma)
    def PY(self, t, xp, x):
        return dists.Normal(scale=np.exp(0.5 * x))

my_sv_model = StochVol(mu=0.8, rho=0.95, sigma=1.)
my_sv_model.mu

x0 = my_sv_model.PX0().rvs(size=30) # generate 30 draws from PX0
x, y = my_sv_model.simulate(100) # simulate (X_t, Y_t) from the model

# plt.plot(y) # plot the simulated data
# plt.show()

# #########
# Chapter 5

class Bootstrap_SV(particles.FeynmanKac):
    """Bootstrap FK model associated to a stochastic volatility ssm. """
    def __init__(self, data=None, mu=0., sigma=1., rho=0.95):
        self.data = data
        self.T = len(data) # number of time steps
        self.mu = mu
        self.sigma = sigma
        self.rho = rho
        self.sigma0 = self.sigma / np.sqrt(1. - self.rho**2)
    def M0(self, N):
        return dists.Normal(loc=self.mu, scale=self.sigma0).rvs(size=N)
    def M(self, t, xp):
        return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), scale=self.sigma).rvs(size=xp.shape[0])
    def logG(self, t, xp, x):
        return dists.Normal(scale=np.exp(0.5 * x)).logpdf(self.data[t])

y = dists.Normal().rvs(size=100) # artificial data
fk_boot_sv = Bootstrap_SV(mu=-1., sigma=0.15, rho=0.9, data=y)

# It would be nice if we could generate automatically the corresponding Bootstrap Feynman-Kac model, without defining it manually, where my_sv_model is the state-space model defined above
fk_boot_sv = ssm.Bootstrap(ssm=my_sv_model, data=y)

# #########
# Chapter 8

# Multiplying two numbers stored on the log-scale is easy: simply add their logs. Adding two such numbers is a bit more tricky.

def logplus(la, lb):
    """Sum of two numbers stored on log-scale."""
    if la > lb:
        return la + np.log1p(np.exp(lb - la)) # log1p(x) = log(1 + x)
    else:
        return lb + np.log1p(np.exp(la - lb))

# How do these remarks apply to importance sampling? importance weights should of course also be computed on the log scale. To obtain normalised weights, one may adapt the logplus function as follows:

def exp_and_normalise(lw):
    """Compute normalised weigths W, from log weights lw."""
    w = np.exp(lw - lw.max())
    return w / w.sum()

# With these remarks in mind, here is a generic function to compute importance sampling estimates:

def importance_sampling(target, proposal, phi, N=1000):
    x = proposal.rvs(size=N)
    lw = target.logpdf(x) - proposal.logpdf(x)
    W = exp_and_normalise(lw)
    return np.average(phi(x), weights=W)

# and here is a quick example:

f = lambda x: x # function f(x)=x
est = importance_sampling(stats.norm(), stats.norm(scale=2.), f)

# #########
# Chapter 9

def inverse_cdf(su, W):
    """Inverse CDF algorithm for a finite distribution.
    Parameters
    ----------
    su: (M,) ndarray
    M sorted variates (i.e. M ordered points in [0,1]).
    W: (N,) ndarray
    a vector of N normalized weights (>=0 and sum to one)
    Returns
    -------
    A: (M,) ndarray
    a vector of M indexes in range 0, ..., N-1
    """
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, 'int')
    for n in range(M):
        while su[n] > s:
            # Si
            j += 1
            s += W[j]
        A[n] = j
    return A


def stratified(M, W):
    su = (random.rand(M) + np.arange(M)) / M
    return inverse_cdf(su, W)

def systematic(M, W):
    su = (random.rand(1) + np.arange(M)) / M
    return inverse_cdf(su, W)

# #########
# Chapter 10


# Object fk_boot_sv was defined in the Python corner of Chap. 5
mypf = particles.SMC(fk=fk_boot_sv, N=100, resampling='systematic', ESSrmin=0.5)
mypf.run()

mypf.X
mypf.W

# How do we specify other particle algorithms such as guided particle filters? First, we need to enrich our state space model with proposal distributions. The class below defines a basic linear Gaussian state-space model, plus the locally optimal proposal at time 0 and at times t ≥ 1

class LinearGauss(ssm.StateSpaceModel):
    default_params = {'sigmaY': .2, 'rho': 0.9, 'sigmaX': 1.}
    def PX0(self): # X_0 ~ N(0, sigmaX^2)
        return dists.Normal(scale=self.sigmaX)
    def PX(self, t, xp): # X_t | X_{t-1} ~ N(rho * X_{t-1}, sigmaX^2)
        return dists.Normal(loc=self.rho * xp, scale=self.sigmaX)
    def PY(self, t, xp, x): # Y_t | X_t ~ N(X_t, sigmaY^2)
        return dists.Normal(loc=x, scale=self.sigmaY)
    def proposal0(self, data):
        sig2post = 1. / (1. / self.sigmaX**2 + 1. / self.sigmaY**2)
        mupost = sig2post * (data[0] / self.sigmaY**2)
        return dists.Normal(loc=mupost, scale=np.sqrt(sig2post))
    def proposal(self, t, xp, data):
        sig2post = 1. / (1. / self.sigmaX**2 + 1. / self.sigmaY**2)
        mupost = sig2post * (self.rho * xp / self.sigmaX**2
        + data[t] / self.sigmaY**2)
        return dists.Normal(loc=mupost, scale=np.sqrt(sig2post))

model = LinearGauss() # default parameters
x, y = model.simulate(100) # simulate data
fk_guided = ssm.GuidedPF(ssm=model, data=y)
pf = particles.SMC(fk=fk_guided, N=1000)
pf.run()

geometric_evidence = np.exp(pf.summaries.logLts[-1]/len(pf.summaries.logLts))

# Package particles provides a function called multiSMC to run several particle filters, optionally in parallel. For instance, this:

results = particles.multiSMC(fk=fk_boot_sv, N=100, nruns=10, nprocs=0)

print(results[0]["output"])







