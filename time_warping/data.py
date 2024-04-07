import lightning as pl
import torch
import pdb
import numpy as np
import math
from utils import create_psi
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import torch.nn.functional as F

class DATASET(Dataset):
    def __init__(self, N,T,nb_basis=64):
        self.X = []
        self.y = []
        self.N = N
        self.T = T
        self.nb_basis = nb_basis
        Z_all = []
        l_all = []
        for i in range(N):
            #using torch and not numpy
            l = torch.rand(1)*20
            Z = torch.rand(4)*8-4
            l_all.append(l)
            Z_all.append(Z)
            X_obs,X_orig = self.generate_trajectory(torch.linspace(0,1,T),l,Z)
            self.X.append(X_obs)
        for i in range(N):
            Z = Z_all[i]
            pk = self.class_probs(Z)
            y = (pk[0]>0.5).float()
            self.y.append(y)
        self.X = torch.stack(self.X)
        self.y = torch.stack(self.y)
        G,F = self.get_G(self.X.shape[1],nb_basis)
        B = self.X.matmul(G)
        self.B = B

    def integral(self,l,t):
        return (1-torch.exp(-l*t))/l

    def g(self,l,t):
        C = l/(1-torch.exp(-l))
        return C*self.integral(l,t)
    
    def g_deriv(self,l,t):
        C = l/(1-torch.exp(-l))
        return C*torch.exp(-l*t)
    
    def sigmoid(self,t):
        return 1 / (1 + torch.exp(-t))
    
    def f(self,t,Z):
        return Z[0]*torch.cos(9*math.pi*t)*(t<0.25)+Z[1]*(t**2)*(t>=0.25)*(t<0.5)+Z[2]*torch.sin(t)*(t>=0.5)*(t<0.75)+Z[3]*(torch.cos(17*math.pi*t))*(t>=0.75)*(t<=1)
    
    def class_probs(self,Z):
        c1 = 2*(Z[0]*math.sin(9*math.pi*0.25)/(9*math.pi)+Z[1]*(0.5**3/3-0.25**3/3))
        c2 = -2*Z[2]*(math.cos(0.75)-math.cos(0.5))+2*Z[3]*(np.sin(17*math.pi)/(17*math.pi)-np.sin(17*math.pi*0.75)/(17*math.pi))
        #return softmax applied to a torch array formed from c1 and c2
        return F.softmax(torch.tensor([c1,c2]))

    def generate_trajectory(self,t,l,Z):
        inv_warp = self.g(l,t)
        X_obs = self.f(inv_warp,Z)
        X_orig = self.f(t,Z)
        return X_obs,X_orig
    
    def p_i(self,l,t,c):
        C = l/(1-torch.exp(-l))
        if c==0:
            return 2*(self.g(l,t)<0.5)*self.g_deriv(l,t)
        else:
            return 2*(self.g(l,t)>=0.5)*self.g_deriv(l,t)
        
    def get_G(self, max_length,nb_basis):
        psis=[]
        Gs = []

        for length in range(1,max_length+1):
            psi = create_psi(length,nb_basis)
            shift = 1 / float(2 * length)
            positions = torch.linspace(shift, 1 - shift, length)
            positions = positions.unsqueeze(1)
            all_basis = [basis_function.evaluate(positions)
                                for basis_function in psi]
            F = torch.cat(all_basis, dim=-1).t()
            nb_basis = sum([len(b) for b in psi])
            assert F.size(0) == nb_basis

            # compute G with a ridge penalty
            penalty = 0.01
            I = torch.eye(nb_basis)
            G = F.t().matmul((F.matmul(F.t()) + penalty * I).inverse())
            psis.append(psi)
            Gs.append(G)
        G = Gs[max_length-1]
        return G,F

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx,:], self.B[idx,:], self.y[idx]
    
if __name__=='__main__':
    X_obs_all = []
    X_orig_all = []
    l_all = []
    Z_all = []
    max_l=25
    max_l=20
    T = 95
    N = 5
    dataset = DATASET(N,T)