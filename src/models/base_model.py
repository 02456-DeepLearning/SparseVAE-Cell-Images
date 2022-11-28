import pdb
import torch
from torchvision.utils import save_image
from pathlib import Path
from glob import glob
import pandas as pd
from tqdm import tqdm

from logger import Logger

class VariationalBaseModel():
    def __init__(self, dataset, width, height, channels, latent_sz, 
                 learning_rate, device, log_interval, normalize=False, 
                 flatten=True):
        self.dataset = dataset
        self.width = width
        self.height = height
        self.channels = channels
        # before width * height * channels
        self.input_sz = channels*width*height
        self.input_sz_tup = (channels,width,height)
        self.latent_sz = latent_sz
        
        self.lr = learning_rate
        self.device = device
        self.log_interval = log_interval
        self.normalize_data = normalize
        self.flatten_data = flatten
        
        # To be implemented by subclasses
        self.model = None
        self.optimizer = None        
    
    
    def loss_function(self):
        raise NotImplementedError
    
    
    def step(self, data, train=False):

        if train:
            self.optimizer.zero_grad()
        output = self.model(data)
        # recon_x, mu, logvar, logspike= output
        #pdb.set_trace()
        loss, log = self.loss_function(data, *output)
 
        if train:
            loss.backward()
            self.optimizer.step()


        return loss.item(), log
    
    # TODO: Perform transformations inside DataLoader (extend datasets.MNIST)
    def transform(self, batch):
        if self.flatten_data: 
            batch_size = len(batch)
            batch = batch.view(batch_size, -1)
        if self.normalize_data:
            batch = batch / self.scaling_factor
#         batch_norm = flattened_batch.norm(dim=1, p=2)
#         flattened_batch /= batch_norm[:, None]
        return batch
        
    def inverse_transform(self, batch):
        return batch * self.scaling_factor \
                if self.normalize_data else batch
    
    def calculate_scaling_factor(self, data_loader):
        print(f'Calculating norm mean of training set')
        norms = []
        self.model.eval()
        n_batches = len(data_loader)
        for batch_idx, (data, _) in enumerate(data_loader):
            batch_size = len(data)
            flattened_batch = data.view(batch_size, -1)
            batch_norm = flattened_batch.norm(dim=1, p=2)
            norms.extend(list(batch_norm.numpy()))
        norms = pd.Series(norms)
        print(norms.describe())
        self.scaling_factor = norms.mean()
        print('Done!\n')
    
    
    # Run training iterations and report results
    def train(self, train_loader, epoch, logging_func=print):
        self.model.train()
        train_loss = 0

        logs = {
            'LOSS': 0,
            'BCE': 0,
            'PRIOR': 0,
            'prior1': 0,
            'prior2': 0
        }
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = self.transform(data).to(self.device)
            loss, log = self.step(data, train=True)
            train_loss += loss
            logs['LOSS'] += log['LOSS']
            logs['BCE'] += log['BCE']
            logs['PRIOR'] += log['PRIOR']
            logs['prior1'] += log['prior1']
            logs['prior2'] += log['prior2']
            if batch_idx % self.log_interval == 0:
                logging_func('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}' \
                      .format(epoch, batch_idx * len(data), 
                              len(train_loader.dataset),
                              100. * batch_idx / len(train_loader),
                              loss / len(data)))

        logging_func('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

        final_loss = train_loss / len(train_loader.dataset)
        logs['LOSS'] /= len(train_loader.dataset)
        logs['BCE'] /= len(train_loader.dataset)
        logs['PRIOR'] /= len(train_loader.dataset)
        logs['prior1'] /= len(train_loader.dataset)
        logs['prior2'] /= len(train_loader.dataset)
        return final_loss, logs
        
        
    # Returns the VLB for the test set
    def test(self, test_loader, epoch, logging_func=print):
        self.model.eval()
        test_loss = 0
        
        logs = {
            'LOSS': 0,
            'BCE': 0,
            'PRIOR': 0,
            'prior1': 0,
            'prior2': 0
        }

        with torch.no_grad():
            for data, _ in test_loader:
                data = self.transform(data).to(self.device)
                loss, log = self.step(data, train=False)
                test_loss += loss
                logs['LOSS'] += log['LOSS']
                logs['BCE'] += log['BCE']
                logs['PRIOR'] += log['PRIOR']
                logs['prior1'] += log['prior1']
                logs['prior2'] += log['prior2']
                
                
        VLB = test_loss / len(test_loader)
        ## Optional to normalize VLB on testset
        name = self.model.__class__.__name__
        test_loss /= len(test_loader.dataset) 
        logging_func(f'====> Test set loss: {test_loss:.4f} - VLB-{name} : {VLB:.4f}')

        logs['LOSS'] /= len(test_loader.dataset) 
        logs['BCE'] /= len(test_loader.dataset) 
        logs['PRIOR'] /= len(test_loader.dataset) 
        logs['prior1'] /= len(test_loader.dataset) 
        logs['prior2'] /= len(test_loader.dataset) 
        return test_loss, logs
    
    
    #Auxiliary function to continue training from last trained models
    def load_last_model(self, checkpoints_path, logging_func=print):
        name = self.model.__class__.__name__
        # Search for all previous checkpoints
        models = glob(f'{checkpoints_path}/*.pth')
        model_ids = []
        for f in models:
            # modelname_dataset_startepoch_epochs_latentsize_lr_epoch
            run_name = Path(f).stem
            model_name, dataset, _, _, latent_sz, _, epoch = run_name.split('_')
            if model_name == name and dataset == self.dataset and \
               int(latent_sz) == self.latent_sz:
                model_ids.append((int(epoch), f))
                
        # If no checkpoints available
        if len(model_ids) == 0:
            logging_func(f'Training {name} model from scratch...')
            return 1

        # Load model from last checkpoint 
        start_epoch, last_checkpoint = max(model_ids, key=lambda item: item[0])
        logging_func('Last checkpoint: ', last_checkpoint)
        self.model.load_state_dict(torch.load(last_checkpoint))
        logging_func(f'Loading {name} model from last checkpoint ({start_epoch})...')

        return start_epoch + 1
    
    
    def update_(self):
        pass
    
    
    def run_training(self, train_loader, test_loader, epochs, 
                     report_interval, sample_sz=64, reload_model=True,
                     checkpoints_path='./results/checkpoints',
                     logs_path='./results/logs',
                     images_path='./results/images',
                     logging_func=print, start_epoch=None):
        
        if self.normalize_data:
            self.calculate_scaling_factor(train_loader)
        
        if start_epoch is None:
            start_epoch = self.load_last_model(checkpoints_path, logging_func) \
                                           if reload_model else 1
        name = self.model.__class__.__name__
        run_name = f'{name}_{self.dataset}_{start_epoch}_{epochs}_' \
                   f'{self.latent_sz}_{str(self.lr).replace(".", "-")}'
        logger = Logger(f'{logs_path}/{run_name}')
        logging_func(f'Training {name} model...')
        for epoch in range(start_epoch, start_epoch + epochs):
            train_loss, logs = self.train(train_loader, epoch, logging_func)
            test_loss, logs = self.test(test_loader, epoch, logging_func)
            # Store log
            # pdb.set_trace()
            logger.scalar_summary(train_loss, test_loss, epoch, logs)
            # Optional update
            #self.update_()
            # For each report interval store model and save images
            if epoch % report_interval == 0:
                with torch.no_grad():

                    ## Generate random samples
                    sample = torch.randn(sample_sz, self.latent_sz) \
                                  .to(self.device)
                    sample = self.model.decode(sample).cpu()
                    sample = self.inverse_transform(sample)
                    ## Store sample plots
                    save_image(sample.view(sample_sz, self.channels, self.height,
                                           self.width),
                               f'{images_path}/sample_{run_name}_{epoch}.png')
                    ## Store Model
                    torch.save(self.model.state_dict(), 
                               f'{checkpoints_path}/{run_name}_{epoch}.pth')
