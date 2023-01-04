################################################################################################################
### Based on the codebase from the ICLR 2019 Reproducibility Challenge entry for "Variational Sparse Coding" ###
#################### Link to repository: https://github.com/Alfo5123/Variational-Sparse-Coding #################
####################### Credits to Alfredo de la Fuente Briceño - See also LICENSE file ########################
################################################################################################################

import torch
import pdb
from utils import get_argparser, get_datasets
from models.conv_vsc import ConvolutionalVariationalSparseCoding
from models.enc_classifier import ClassifierModelFull

if __name__ == "__main__":    
    parser = get_argparser('ConvVSC Example')
    parser.add_argument('--alpha', default=0.2, type=float, metavar='A', 
                    help='value of spike variable (default: 0.2')
    parser.add_argument('--beta-delta', default=0, type=float, metavar='betadelta', 
                    help='beta delta (default: 0')
    parser.add_argument('--kernel-size', type=str, default='32,32,68,68', metavar='HS',
                        help='kernel sizes, separated by commas (default: 32,32,68,68)')
    parser.add_argument('--fold-number', type=int, default=1, metavar='FN',
                        help='what fold to use for test data')
    args = parser.parse_args()
    print('ConvVSC Baseline Experiments\n')
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
                                                                        args.cuda,fold_number=args.fold_number)
    
    # Tune the learning rate (All training rates used were between 0.001 and 0.01)
    vsc = ConvolutionalVariationalSparseCoding(args.dataset, width, height, channels, 
                                  args.kernel_size, args.hidden_size, args.latent_size, 
                                  args.lr, args.alpha, device, args.log_interval,
                                  args.normalize,flatten=False, model_type=f"convvsc_{args.fold_number}", beta_delta=args.beta_delta)
    vsc.run_training(train_loader, test_loader, args.epochs,
                     args.report_interval, args.sample_size, 
                     reload_model=not args.do_not_resume)
    