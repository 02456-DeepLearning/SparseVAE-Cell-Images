import pdb

import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from utils import get_datasets

from models.conv_vsc import ConvVSC
from models.conv_vae import ConvVAE

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
                 hl_size_1, hl_size_2, learning_rate, alpha,
                 device, log_interval, normalize, flatten, **kwargs):
        hidden_size, latent_size = hl_size_1 
        hidden_size_2, latent_size_2 = hl_size_2
        super().__init__(dataset, width, height, channels, latent_size,
                         learning_rate, device, log_interval, normalize, 
                         flatten)

        self.alpha = alpha
        self.hidden_sz = int(hidden_size)
        self.kernel_szs = [int(ks) for ks in str(kernel_szs).split(',')]
        self.conv_vsc = ConvVSC(self.input_sz_tup, self.kernel_szs, hidden_size,
                             latent_size, **kwargs).to(device)
        self.conv_vae = ConvVAE(self.input_sz_tup, self.kernel_szs, hidden_size_2,
                             latent_size_2, **kwargs).to(device)
        
        # MODEL_PATH = '/zhome/2b/d/156632/Desktop/deeplearning/Variational-Sparse-Coding/results_shared/checkpoints/convvsc_1_ConvVSC_stratifiedcell_1_300_30_0-001_300.pth'
        MODEL_PATH = '/zhome/2b/d/156632/Desktop/deeplearning/Variational-Sparse-Coding/results_shared/checkpoints/convvsc_5_ConvVSC_stratifiedcell_1_75_200_0-001_75.pth'
        MODEL_PATH_2 = '/zhome/2b/d/156632/Desktop/deeplearning/Variational-Sparse-Coding/results_shared/checkpoints/convvae_5_ConvVAE_stratifiedcell_1_75_200_0-001_75.pth'
        self.load_specific_model(self.conv_vsc, MODEL_PATH)
        self.load_specific_model(self.conv_vae, MODEL_PATH_2)
        self.model = VisualizerUtilModel(self.conv_vsc, 13).to(device)
        self.model_2 = VisualizerUtilModel(self.conv_vae, 13).to(device)
    
    #Auxiliary function to continue training from last trained models
    def load_specific_model(self, conv_vsc, model_path, logging_func=print):
        conv_vsc.load_state_dict(torch.load(model_path))
        logging_func(f'Loading trained model from {model_path}')

        return 0
    
    def encode_and_write(self, batch, labels):
        z,latent_reps,_,_ = self.conv_vsc.forward(batch)
        latent_reps = latent_reps.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        # write embeddings to .tsv
        df = pd.DataFrame(latent_reps)
        df.to_csv('tsne-data/embeddings.tsv', mode='a', sep='\t',header=False,index=False)
        # write true labels to .tsv
        df = pd.DataFrame(labels)
        df.to_csv('tsne-data/labels.tsv', mode='a', sep='\t',header=False,index=False)

    def decode_and_write(self, batch):
        f, axarr = plt.subplots(3,3)

        reconstruction,_,_,_ = self.conv_vsc.forward(batch)
        reconstruction = np.moveaxis(reconstruction.cpu().numpy(),[0,1,2,3],[0,3,2,1])
        img_0_recon = reconstruction[0:3,:,:,:]

        reconstruction_2,_,_ = self.conv_vae.forward(batch)
        reconstruction_2 = np.moveaxis(reconstruction_2.cpu().numpy(),[0,1,2,3],[0,3,2,1])
        img_0_recon_2 = reconstruction_2[0:3,:,:,:]

        imgs = np.moveaxis(batch.cpu().numpy(),[0,1,2,3],[0,3,2,1])
        img_0 = imgs[0:3,:,:,:]

        # plt.imsave("test.png", batch[])
        # plt.imsave("test_recon.png", img_0/img_0.max(axis=(0,1)))

        for i in range(3):
            axarr[i,0].imshow(img_0[i]/img_0[i].max(axis=(0,1)))
            axarr[i,1].imshow(img_0_recon[i]/img_0_recon[i].max(axis=(0,1)))
            axarr[i,2].imshow(img_0_recon_2[i]/img_0_recon_2[i].max(axis=(0,1)))
        plt.savefig("test_recon.png")

        # pdb.set_trace()




if __name__ == "__main__":    
    # which visualization operation to run
    # ["embed","{{START}}","{{END}}"] => write .tsv files with embeddings and labels for batch_idx in range (START, END)
    # ["recon","{{START}}","{{END}}"] => plot original images next to reconstruction
    # ""
    viz_type = sys.argv[1]
    START, END = (int(sys.argv[2]), int(sys.argv[3]))
    
    dataset = "cell"
    epochs = 200
    kernel_size = '32,32,68,68' # parameters

    # SCVAE parameters
    hidden_size = 400
    latent_size = 200
    hl_size_1 = (hidden_size, latent_size)

    # CVAE parameters
    hidden_size_2 = 400
    latent_size_2 = 200
    hl_size_2 = (hidden_size_2, latent_size_2)

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
                                  kernel_size, hl_size_1, hl_size_2, learning_rate,
                                  alpha, device, log_interval, normalize, flatten)
    
    #Set reproducibility seed
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if viz_type=="embed":
        for batch_idx, (batch,y) in enumerate(train_loader,START):
            if batch_idx > END:
                break
            data = visualizer.transform(batch).to(visualizer.device)
            # pdb.set_trace()
            visualizer.encode_and_write(data,y)
            # pdb.set_trace()
    elif viz_type=="recon":
        for batch_idx, (batch,y) in enumerate(train_loader,START):
            if batch_idx > END:
                break
            data = visualizer.transform(batch).to(visualizer.device)
            # pdb.set_trace()
            visualizer.decode_and_write(data)