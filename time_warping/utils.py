import torch
import math
from basis_functions import GaussianBasisFunctions

def add_gaussian_basis_functions(nb_basis, sigmas):
    mu, sigma = torch.meshgrid(torch.linspace(0, 1, nb_basis // len(sigmas)),
                               torch.Tensor(sigmas))
    mus = mu.flatten()
    sigmas = sigma.flatten()
    return GaussianBasisFunctions(mus, sigmas)

def create_psi(length,nb_basis):
    psi = []
    nb_waves = nb_basis
    nb_waves = max(2,nb_waves)
    psi.append(
                add_gaussian_basis_functions(nb_waves,
                                             sigmas=[.1, .5],
                                             # sigmas=[.03, .1, .3],
                                            )
            )
    return psi

def _phi(t):
        '''
        @summary: Gaussian radial basis function
        '''
        return 1.0/math.sqrt(2*math.pi)*torch.exp(-0.5*t**2)

def exp_kernel(t,bandwidth):
    '''
    @summary: Exponential kernel
    '''
    return torch.exp(-torch.abs(t)/bandwidth)


def beta_exp(t):
    '''
    @summary: beta-exponential function
    '''
    q = 0
    plus = torch.nn.ReLU()
    return plus(1+(1-q)*t)**(1./(1-q))


def truncated_parabola(t,mu,sigma_sq):
    '''
    @summary: Truncated parabola function
    @param t: torch.Tensor, input
    @param mu: torch.Tensor, mean
    @param sigma_sq: torch.Tensor, variance
    '''
    plus = torch.nn.ReLU()
    return plus(-(t-mu)**2/(2*sigma_sq)+0.5*(3/(2*torch.sqrt(sigma_sq)))**(2./3.))
