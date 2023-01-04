################################################################################################################
### Based on the codebase from the ICLR 2019 Reproducibility Challenge entry for "Variational Sparse Coding" ###
#################### Link to repository: https://github.com/Alfo5123/Variational-Sparse-Coding #################
####################### Credits to Alfredo de la Fuente Briceño - See also LICENSE file ########################
################################################################################################################

import torch

from utils import get_argparser, get_datasets
from models.vsc import VariationalSparseCoding

if __name__ == "__main__":    
    parser = get_argparser('VSC Example')
    parser.add_argument('--alpha', default=0.5, type=float, metavar='A', 
                    help='value of spike variable (default: 0.5')
    args = parser.parse_args()
    print('VSC Baseline Experiments\n')
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #Set reproducibility seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #Define device for training
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(f'Using {device} device...')
    
    #Load datasets
    train_loader, test_loader, (width, height, channels) = get_datasets(args.dataset,
                                                                        args.batch_size,
                                                                        args.cuda)
    
    # Tune the learning rate (All training rates used were between 0.001 and 0.01)
    vsc = VariationalSparseCoding(args.dataset, width, height, channels, 
                                  args.hidden_size, args.latent_size, args.lr, 
                                  args.alpha, device, args.log_interval,
                                  args.normalize, flatten=True)
    vsc.run_training(train_loader, test_loader, args.epochs,
                     args.report_interval, args.sample_size, 
                     reload_model=not args.do_not_resume)
    