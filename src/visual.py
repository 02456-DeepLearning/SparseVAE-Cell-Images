import numpy as np
import matplotlib.pyplot as plt
import torch

import os
import tensorflow as tf
#import tensorflow_datasets as tfds

from tensorboard.plugins import projector

from utils import get_argparser, get_datasets
from models.conv_vsc import ConvolutionalVariationalSparseCoding

    
parser = get_argparser('VSC Example')
parser.add_argument('--alpha', default=0.5, type=float, metavar='A', 
                help='value of spike variable (default: 0.5')
args = parser.parse_args()



print('VSC Baseline Experiments\n')
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.dataset = 'cell'
args.latent_size=200
args.hidden_size=400
args.normalize=False

device = torch.device('cuda' if args.cuda else 'cpu')
#Set reproducibility seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

train_loader, test_loader, (width, height, channels) = get_datasets(args.dataset,
                                                                        args.batch_size,
                                                                        args.cuda)

vsc = ConvolutionalVariationalSparseCoding(dataset=args.dataset, width=width, height=height, channels=channels, kernel_szs="32, 32, 68, 68", 
                                  hidden_sz=args.hidden_size, latent_sz=args.latent_size, learning_rate=args.lr, 
                                  alpha=args.alpha, device=device, log_interval=args.log_interval,
                                  normalize=False, flatten=True)

#vsc.load_last_model('./results/checkpoints')
chkpt='./results/checkpoints/ConvVSC_cell_1_481_200_0-001_420.pth'
vsc.model.load_state_dict(torch.load(chkpt))
vsc.model.to(device)
vsc.model.eval()
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        mu, logvar, logspike = vsc.model.encode(data)
        encoded = vsc.model.reparameterize(mu, logvar, logspike)
        break
print(encoded)



config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = encoded.name



projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)