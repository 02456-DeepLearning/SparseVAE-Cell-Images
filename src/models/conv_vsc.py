################################################################################################################
### Based on the codebase from the ICLR 2019 Reproducibility Challenge entry for "Variational Sparse Coding" ###
#################### Link to repository: https://github.com/Alfo5123/Variational-Sparse-Coding #################
####################### Credits to Alfredo de la Fuente Briceño - See also LICENSE file ########################
################################################################################################################

from typing import List, Tuple
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import pdb
from .base_model import VariationalBaseModel


# Convolutional Variational Sparse Coding Model 
class ConvVSC(nn.Module):
    
    def __init__(self, input_sz: Tuple[int, int, int] = (3, 68, 68), 
                 kernel_szs: List[int] = [32, 32, 68, 68], 
                 hidden_sz: int = 256,
                 latent_sz: int = 32,
                 c: float = 50,
                 c_delta: float = 0.001,
                 beta: float = 0.1,
                 beta_delta: float = 0):
        
        super(ConvVSC, self).__init__()
        self.input_sz_tup = input_sz

        self.channel_szs = [input_sz[0]] + kernel_szs 
        self.hidden_sz = hidden_sz
        self.latent_sz = latent_sz
        self.c = c
        self.c_delta = c_delta
        self.beta = beta
        self.beta_delta = beta_delta
        
        conv_modules = [(
            nn.Conv2d(self.channel_szs[i], self.channel_szs[i+1], 
                      (4, 4), stride=2, padding=1),
            nn.ReLU()
            ) for i in range(len(kernel_szs))
        ]
        
        self.conv_encoder = nn.Sequential(*[
            layer for module in conv_modules for layer in module
        ])
        
        conv_out_channels = int(input_sz[-1] / (2 ** len(kernel_szs)))
        self.conv_output_sz = (self.channel_szs[-1], conv_out_channels, 
                               conv_out_channels)
        self.flat_conv_output_sz = np.prod(self.conv_output_sz)
        
        self.features_to_hidden = nn.Sequential(
            nn.Linear(self.flat_conv_output_sz, hidden_sz),
            nn.ReLU()
        )
        
        self.fc_mean = nn.Linear(hidden_sz, latent_sz)
        self.fc_logvar = nn.Linear(hidden_sz, latent_sz)
        self.fc_logspike = nn.Linear(hidden_sz, latent_sz)
        
        self.latent_to_features = nn.Sequential(
            nn.Linear(self.latent_sz, self.hidden_sz), nn.ReLU(),
            nn.Linear(self.hidden_sz, self.flat_conv_output_sz), nn.ReLU()
        )
        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        strides = [2,2,2,2]
        paddings = [1,1,1,1]
        deconv_modules = [(
            nn.ConvTranspose2d(self.channel_szs[-i-1], 
                               self.channel_szs[-i-2],
                               (8, 8) if i==len(kernel_szs)-1 else (4, 4),
                               stride=strides[i], padding=paddings[i]
                            ),
            nn.ReLU() if i < len(kernel_szs) - 1 else nn.Sigmoid()
            ) for i in range(len(kernel_szs))
        ]
        print(self.channel_szs,"********")
        self.conv_decoder = nn.Sequential(*[ 
            layer for module in deconv_modules for layer in module
        ])


    def encode(self, x):
        # Recognition function
        # x shape: (batch_sz, n_channels, width)
        features = self.conv_encoder(x)
        features = features.view(-1, self.flat_conv_output_sz)
        hidden = self.features_to_hidden(features)
        return self.fc_mean(hidden), self.fc_logvar(hidden), \
               -F.relu(-self.fc_logspike(hidden))

    def reparameterize(self, mu, logvar, logspike):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = eps.mul(std).add_(mu)
        eta = torch.rand_like(std)
        selection = F.sigmoid(self.c*(eta + logspike.exp() - 1))
        return selection.mul(gaussian)

    def decode(self, z):
        #Likelihood function
        features = self.latent_to_features(z)
        #pdb.set_trace()
        features = features.view(-1, *self.conv_output_sz)
        y = self.conv_decoder(features)
        #pdb.set_trace()
        return y

    def forward(self, x):
        # pdb.set_trace()
        mu, logvar, logspike = self.encode(x)
        z = self.reparameterize(mu, logvar, logspike)
        return self.decode(z), mu, logvar, logspike
    
    def update_c(self):
        # Gradually increase c
        self.c += self.c_delta  
    
    def update_beta(self):
        # Gradually adjust beta
        self.beta += self.beta_delta

    
class ConvolutionalVariationalSparseCoding(VariationalBaseModel):
    def __init__(self, dataset, width, height, channels, kernel_szs,
                 hidden_sz, latent_sz, learning_rate, alpha,
                 device, log_interval, normalize, flatten, model_type = "SCVAE",beta_delta=0, **kwargs):
        super().__init__(dataset, width, height, channels, latent_sz,
                         learning_rate, device, log_interval,model_type, normalize, 
                         flatten)
        self.alpha = alpha
        self.hidden_sz = int(hidden_sz)
        self.kernel_szs = [int(ks) for ks in str(kernel_szs).split(',')]

        self.model = ConvVSC(self.input_sz_tup, self.kernel_szs, self.hidden_sz,
                             latent_sz,beta_delta=beta_delta, **kwargs).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_losses = []
        self.test_losses = []
    
        print(self.model)
    
    # Reconstruction + KL divergence losses summed over all elements of batch
    def loss_function(self, x, recon_x, mu, logvar, logspike, train=False):
        # Reconstruction term sum (mean?) per batch
        flat_input_sz = np.prod(self.input_sz_tup)
        #pdb.set_trace()
        BCE = F.binary_cross_entropy(recon_x.view(-1, flat_input_sz), 
                                     x.view(-1, flat_input_sz),
                                     size_average = False)
        # see Appendix B from VSC paper / Formula 6
        spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6) 

        prior1 = -0.5 * torch.sum(spike.mul(1 + logvar - mu.pow(2) - logvar.exp()))
        prior21 = (1 - spike).mul(torch.log((1 - spike) / (1 - self.alpha)))
        prior22 = spike.mul(torch.log(spike / self.alpha))
        prior2 = torch.sum(prior21 + prior22)
        PRIOR = prior1 + prior2

        LOSS = BCE + self.model.beta * PRIOR
        log = {
            'LOSS': LOSS.item(),
            'BCE': BCE.item(),
            'PRIOR': PRIOR.item(),
            'prior1': prior1.item(),
            'prior2': prior2.item(),
            'c': self.model.c,
            'beta': self.model.beta,
            'alpha': self.alpha
        }

        if train:
            self.train_losses.append(log)
        else:
            self.test_losses.append(log)

        return LOSS, log
    
    
    def update_(self):
        # Update value of c gradually 200 ( 150 / 20K = 0.0075 )
        print('updated c and beta', self.model.c,self.model.beta)
        self.model.update_c()
        self.model.update_beta()
        
