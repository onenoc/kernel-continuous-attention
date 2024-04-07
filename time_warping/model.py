import lightning as pl
import torch
from torch import nn
from utils import add_gaussian_basis_functions, create_psi, _phi, exp_kernel, beta_exp, truncated_parabola
import math
import pdb
from lightning.pytorch.utilities import grad_norm


class FeedforwardEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        #add dimensions as attributes of class
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        #define the encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class Attention(torch.nn.Module):
    def __init__(self,input_dim, heads,nb_basis,inducing_points,method='cts_softmax'):
        super(Attention, self).__init__()
        '''
        @param input_dim: int, dimension of input, which is actually the output dimension of the encoder
        @param heads: int, number of heads
        @param nb_basis: int, number of basis functions
        @param inducing_points: int, number of inducing points
        @param method: str, method to use for attention
        '''
        self.heads = heads
        self.inducing_points = inducing_points
        self.method = method
        self.inducing_locations = torch.linspace(0,1,inducing_points)
        #map input to mu, sigma_sq, alpha 
        self.encode_mu = torch.nn.Linear(input_dim, heads)
        self.encode_sigma_sq1 = torch.nn.Linear(input_dim, heads)
        self.encode_sigma_sq2 = torch.nn.Softplus()
        self.encode_alpha = torch.nn.Linear(input_dim,heads*inducing_points)
        self.attn_weights = torch.nn.Softmax(1)
        #add basis functions
        GB = add_gaussian_basis_functions(nb_basis,
                                            sigmas=[.1, .5])
        #add basis functions as attributes of class
        self.mu_basis = GB.mu.unsqueeze(0)
        self.sigma_basis = GB.sigma.unsqueeze(0)
        #initialize a,b,dx,bandwidth
        self.a=0
        self.b=1
        self.dx = 100
        self.bandwidth = 0.05
 
    def _integrate_wrt_kernel_deformed(self,mu,sigma_sq,alpha):
        T = torch.linspace(self.a,self.b,self.dx).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #phi1_upper: size 1 x 1 x nb x dx
        phi1_upper = self.mu_basis.unsqueeze(-1)-T
        #phi1_lower: size 1 x 1 x nb x 1
        phi1_lower = self.sigma_basis.unsqueeze(-1)
        #phi1: size 1 x 1 x nb x dx
        phi1 = _phi(phi1_upper/phi1_lower)/phi1_lower
        K_inputs = torch.cdist(self.inducing_locations.unsqueeze(-1),torch.linspace(self.a,self.b,self.dx).unsqueeze(-1))
        K = exp_kernel(K_inputs,self.bandwidth)
        f = torch.matmul(alpha,K).unsqueeze(-2)#-0.5*(mu.unsqueeze(-1).unsqueeze(-1)-T)**2/sigma_sq.unsqueeze(-1).unsqueeze(-1)
        f_max = torch.max(f,dim=-1,keepdim=True).values
        exp_terms = beta_exp(f-f_max)
        Z = torch.trapz(exp_terms,torch.linspace(self.a,self.b,self.dx),dim=-1).unsqueeze(-1)
        numerical_integral = torch.trapz(phi1*exp_terms/Z,torch.linspace(self.a,self.b,self.dx),dim=-1)
        return numerical_integral
    
    def _integrate_wrt_truncated_parabaloid(self,mu,sigma_sq):
        T = torch.linspace(self.a,self.b,self.dx).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #phi1_upper: size 1 x 1 x nb x dx
        phi1_upper = self.mu_basis.unsqueeze(-1)-T
        #phi1_lower: size 1 x 1 x nb x 1
        phi1_lower = self.sigma_basis.unsqueeze(-1)
        #phi1: size 1 x 1 x nb x dx
        phi1 = _phi(phi1_upper/phi1_lower)/phi1_lower
        deformed_term = truncated_parabola(T,mu.unsqueeze(-1).unsqueeze(-1),sigma_sq.unsqueeze(-1).unsqueeze(-1))
        unnormalized_density = deformed_term
        Z = torch.trapz(unnormalized_density,torch.linspace(self.a,self.b,self.dx),dim=-1).unsqueeze(-1)
        numerical_integral = torch.trapz(phi1*unnormalized_density/Z,torch.linspace(self.a,self.b,self.dx),dim=-1)
        return numerical_integral
    
    def _integrate_product_of_gaussians(self,mu,sigma_sq):
        #T: size 1 x 1 x 1 x dx
        T = torch.linspace(self.a,self.b,self.dx).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #phi1_upper: size 1 x 1 x nb x dx
        phi1_upper = self.mu_basis.unsqueeze(-1)-T
        #phi1_lower: size 1 x 1 x nb x 1
        phi1_lower = self.sigma_basis.unsqueeze(-1)
        #phi1: size 1 x 1 x nb x dx
        phi1 = _phi(phi1_upper/phi1_lower)/phi1_lower
        #phi2_upper: size bs x heads x 1 x dx
        phi2_upper = mu.unsqueeze(-1).unsqueeze(-1)-T
        #phi2_lower: size bs x heads x 1 x 1
        phi2_lower = sigma_sq.unsqueeze(-1).unsqueeze(-1).pow(0.5)
        #phi2: size bs x heads x 1 x dx
        phi2 = _phi(phi2_upper/phi2_lower)/phi2_lower
        #phi1*phi2: size bs x heads x nb x dx
        numerical_integral = torch.trapz(phi1*phi2,torch.linspace(self.a,self.b,self.dx),dim=-1)           
        return numerical_integral
    
    def _integrate_kernel_exp_wrt_gaussian(self,mu,sigma_sq,alpha):
        #T: size 1 x 1 x 1 x dx
        T = torch.linspace(self.a,self.b,self.dx).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #K: kernel matrix, inducing x dx
        K_inputs = torch.cdist(self.inducing_locations.unsqueeze(-1),torch.linspace(self.a,self.b,self.dx).unsqueeze(-1))
        K = exp_kernel(K_inputs,self.bandwidth)
        #f: score, bs x heads x 1 x dx
        f = torch.matmul(alpha,K).unsqueeze(-2)
        #get max of f across dx
        f_max = torch.max(f,dim=-1,keepdim=True).values
        exp_f = torch.exp(f-f_max)
        
        #phi1_upper: size 1 x 1 x nb x dx
        phi1_upper = self.mu_basis.unsqueeze(-1)-T
        #phi1_lower: size 1 x 1 x nb x 1
        phi1_lower = self.sigma_basis.unsqueeze(-1)
        #phi1: size 1 x 1 x nb x dx
        phi1 = _phi(phi1_upper/phi1_lower)/phi1_lower
        #phi2_upper: size bs x heads x 1 x dx
        phi2_upper = mu.unsqueeze(-1).unsqueeze(-1)-T
        #phi2_lower: size bs x heads x 1 x 1
        phi2_lower = sigma_sq.unsqueeze(-1).unsqueeze(-1).pow(0.5)
        #phi2: size bs x heads x 1 x dx
        phi2 = _phi(phi2_upper/phi2_lower)/phi2_lower
        unnormalized_density = phi2*exp_f
        Z = torch.trapz(unnormalized_density,torch.linspace(self.a,self.b,self.dx),dim=-1).unsqueeze(-1)
        #phi1*phi2: size bs x heads x nb x dx
        numerical_integral = torch.trapz(phi1*unnormalized_density/Z,torch.linspace(self.a,self.b,self.dx),dim=-1)      
        return numerical_integral   
    
    def forward(self,x,B):
        v = torch.nan_to_num(x,0)
        #Compute mu, sigma_sq
        mu = self.encode_mu(v)
        sigma_sq = self.encode_sigma_sq1(v)
        sigma_sq = self.encode_sigma_sq2(sigma_sq)
        #alpha: bs x heads x inducing_points
        alpha_init = self.encode_alpha(v)
        alpha = alpha_init.reshape((alpha_init.shape[0],self.heads,self.inducing_points))
        if self.method=='kernel_softmax':
            integrals = self._integrate_kernel_exp_wrt_gaussian(mu,sigma_sq,alpha)
        elif self.method=='cts_softmax':
            integrals = self._integrate_product_of_gaussians(mu,sigma_sq)
        elif self.method=='cts_sparsemax':
            integrals = self._integrate_wrt_truncated_parabaloid(mu,sigma_sq)
        elif self.method=='kernel_sparsemax':
            integrals = self._integrate_wrt_kernel_deformed(mu,sigma_sq,alpha)
        integrals = torch.nan_to_num(integrals,0)
        c = torch.bmm(integrals,B.unsqueeze(-1)).squeeze(-1)
        return c,mu,sigma_sq,alpha
    
class MODEL(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, heads, nb_basis, inducing_points, method, num_classes,optimizer='Adam',lr=1e-4,scheduler=None):
        super().__init__()
        self.encoder = FeedforwardEncoder(input_dim, hidden_dim, output_dim)
        self.attention = Attention(output_dim, heads, nb_basis, inducing_points, method)
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler
    
    def forward(self, x, B):
        x = self.encoder(x)
        c,mu,sigma_sq,alpha = self.attention(x,B)
        return c
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.encoder, norm_type=2)
        self.log_dict(norms)
        norms = grad_norm(self.attention, norm_type=2)
        self.log_dict(norms)

    def training_step(self, batch, batch_idx):
        x,B,y = batch
        y_orig = y.clone()
        #map y to one hot encoding
        y = torch.nn.functional.one_hot(y.to(torch.int64),2).float()
        y_hat = self(x,B)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat,y)+1e-5*torch.norm(y_hat)
        self.log("train_loss", loss, prog_bar=True)
        self.log('train output norm',torch.norm(y_hat),prog_bar=True)
        #compute accuracy
        y_hat = torch.nn.functional.sigmoid(y_hat)
        y_hat = torch.argmax(y_hat,dim=1)
        acc = torch.sum(y_hat==y_orig)/len(y_orig)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,B,y = batch
        y_orig = y.clone()
        #map y to one hot encoding
        y = torch.nn.functional.one_hot(y.to(torch.int64),2).float()
        y_hat = self(x,B)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat,y)
        self.log("val_loss", loss, prog_bar=True)
        #compute accuracy
        y_hat = torch.nn.functional.sigmoid(y_hat)
        y_hat = torch.argmax(y_hat,dim=1)
        acc = torch.sum(y_hat==y_orig)/len(y)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x,B,y = batch
        y_orig = y.clone()
        #map y to one hot encoding
        y = torch.nn.functional.one_hot(y.to(torch.int64),2).float()
        y_hat = self(x,B)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat,y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        #compute accuracy
        y_hat = torch.nn.functional.sigmoid(y_hat)
        y_hat = torch.argmax(y_hat,dim=1)
        acc = torch.sum(y_hat==y_orig)/len(y_orig)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        if self.optimizer=='Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer=='RAdam':
            optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr)
        elif self.optimizer=='SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        if self.scheduler=='StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.scheduler=='ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,verbose=True)
        elif self.scheduler=='CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        if self.scheduler == None:
            return [optimizer]
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


if __name__=='__main__':
    print('hello world')
    