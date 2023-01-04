################################################################################################################
### Based on the codebase from the ICLR 2019 Reproducibility Challenge entry for "Variational Sparse Coding" ###
#################### Link to repository: https://github.com/Alfo5123/Variational-Sparse-Coding #################
####################### Credits to Alfredo de la Fuente Briceño - See also LICENSE file ########################
################################################################################################################

import pdb

import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import pandas as pd

from utils import get_datasets

from models.conv_vsc import ConvVSC

from models.base_model import VariationalBaseModel

# Variational AutoEncoder Model 
class VisualizerUtilModel(nn.Module):
    def __init__(self, conv_vsc, num_classes):
        super(VisualizerUtilModel, self).__init__()
        self.conv_vsc = conv_vsc
        self.num_classes = num_classes

        for p in self.conv_vsc.parameters():
            p.requires_grad = False

        self.flatten = nn.Flatten()

      
    def forward(self, x):
        # pdb.set_trace()
        mu, _ = self.conv_vsc.encode(x)
        
        x = self.flatten(mu)
        return x

    
class VisualizerUtilModelFull(VariationalBaseModel):
    def __init__(self, dataset, width, height, channels, kernel_szs,
                 hidden_sz, latent_sz, learning_rate, alpha,
                 device, log_interval, normalize, flatten, **kwargs):
        super().__init__(dataset, width, height, channels, latent_sz,
                         learning_rate, device, log_interval, normalize, 
                         flatten)

        self.alpha = alpha
        self.hidden_sz = int(hidden_sz)
        self.kernel_szs = [int(ks) for ks in str(kernel_szs).split(',')]
        self.conv_vsc = ConvVSC(self.input_sz_tup, self.kernel_szs, self.hidden_sz,
                             latent_sz, **kwargs).to(device)
        self.load_specific_model('/zhome/2b/d/156632/Desktop/deeplearning/Variational-Sparse-Coding/results/checkpoints/ConvVSC_cell_1_481_200_0-001_420.pth')
        self.model = VisualizerUtilModel(self.conv_vsc, 13).to(device)
    
    #Auxiliary function to continue training from last trained models
    def load_specific_model(self, model_path, logging_func=print):
        self.conv_vsc.load_state_dict(torch.load(model_path))
        logging_func(f'Loading trained model from {model_path}')

        return 0
    
    def encode_and_write(self, batch):
        _,latent_reps,_,_ = self.conv_vsc.forward(batch)
        latent_reps = latent_reps.detach().cpu().numpy()
        # pdb.set_trace()
        df = pd.DataFrame(latent_reps)
        df.to_csv('test.csv')



if __name__ == "__main__":    
    dataset = "cell"
    epochs = 200
    kernel_size = '32,32,68,68' # parameters
    hidden_size = 400
    latent_size = 200
    learning_rate = 0.001
    device = torch.device('cuda')
    alpha = 0.5 ## DEFAULT
    log_interval = -1
    normalize = False ## DEFAULT
    flatten = False ## DEFAULT

    #Load datasets
    batch_size = 32
    train_loader, test_loader, (width, height, channels) = get_datasets(dataset,
                                                                        batch_size,
                                                                        device)

    visualizer = VisualizerUtilModelFull(dataset, width, height, channels, 
                                  kernel_size, hidden_size, latent_size, learning_rate,
                                  alpha, device, log_interval, normalize, flatten)
    # self, dataset, width, height, channels, kernel_szs,
    #              hidden_sz, latent_sz, learning_rate, alpha,
    #              device, log_interval, normalize, flatten, **kwargs):
    #Set reproducibility seed
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

    idx = 0
    # for batch_idx, (data, _) in enumerate(train_loader):
    for batch_idx, (batch,y) in enumerate(train_loader):
        data = visualizer.transform(batch).to(visualizer.device)
        # pdb.set_trace()
        if idx > 0:
            break
        visualizer.encode_and_write(data)
        idx += 1
        # pdb.set_trace()
    