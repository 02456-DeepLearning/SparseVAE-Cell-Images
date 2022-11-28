import pdb

import torch
from torch import nn, optim
from torch.nn import functional as F
from numpy import mean
from sklearn import metrics
import numpy as np

from models.conv_vsc import ConvVSC

from .base_model import VariationalBaseModel

# Variational AutoEncoder Model 
class ClassifierModel(nn.Module):
    def __init__(self, conv_vsc, num_classes):
        super(ClassifierModel, self).__init__()
        self.conv_vsc = conv_vsc
        self.num_classes = num_classes

        for p in self.conv_vsc.parameters():
            p.requires_grad = False

        self.flatten = nn.Flatten()
        self.linear_final = nn.Linear(self.conv_vsc.latent_sz,self.num_classes)

      
    def forward(self, x):
        # pdb.set_trace()
        mu, logvar, logspike = self.conv_vsc.encode(x)
        
        x = self.flatten(mu)
        x = self.linear_final(x)
        return x

    
class ClassifierModelFull(VariationalBaseModel):
    def __init__(self, dataset, width, height, channels, kernel_szs,
                 hidden_sz, latent_sz, learning_rate, alpha,
                 device, log_interval, normalize, flatten, **kwargs):
        super().__init__(dataset, width, height, channels, latent_sz,
                         learning_rate, device, log_interval, normalize, 
                         flatten)

        self.loss = nn.CrossEntropyLoss()        
        self.alpha = alpha
        self.hidden_sz = int(hidden_sz)
        self.kernel_szs = [int(ks) for ks in str(kernel_szs).split(',')]
        self.conv_vsc = ConvVSC(self.input_sz_tup, self.kernel_szs, self.hidden_sz,
                             latent_sz, **kwargs).to(device)
        self.load_specific_model('/zhome/a2/4/155672/Desktop/DeepLearning/SparseVAE-Cell-Images/results/checkpoints/ConvVSC_cell_1_500_204_0-0001_12.pth')
        self.model = ClassifierModel(self.conv_vsc, 13).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    #Auxiliary function to continue training from last trained models
    def load_specific_model(self, model_path, logging_func=print):
        self.conv_vsc.load_state_dict(torch.load(model_path))
        logging_func(f'Loading trained model from {model_path}')

        return 0


    # Reconstruction + KL divergence losses summed over all elements of batch
    def loss_function(self, x, y):
        #pdb.set_trace()
        #print(mean([z.argmax() for z in x]==y))
        #print(self.accuracy(y,[z.argmax() for z in x]))
        return self.loss(x,y)

    def accuracy(self, target, pred):
        # pdb.set_trace()
        pred = np.array([ts.detach().cpu().numpy() for ts in pred])
        return metrics.accuracy_score(target.detach().cpu().numpy(), 
                                        pred)
        