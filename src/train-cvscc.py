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
    parser.add_argument('--kernel-size', type=str, default='32,32,68,68', metavar='HS',
                        help='kernel sizes, separated by commas (default: 32,32,68,68)')
    parser.add_argument('--train-encoder', action='store_true', default=False,
                        help='should the encoder be trained')
    parser.add_argument('--use-vae-encoder', action='store_true', default=False,
                        help='use vae encoder')
    parser.add_argument('--fold-number', type=int, default=1, metavar='FN',
                        help='what fold to use for test data')
    parser.add_argument('--encoder-model-path', type=str, default='32,32,68,68', 
                        help='kernel sizes, separated by commas (default: 32,32,68,68)')


    args = parser.parse_args()
    print('ConvVSC classifier\n')
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
                                                                        args.cuda,
                                                                        fold_number=args.fold_number)
    
    # Tune the learning rate (All training rates used were between 0.001 and 0.01)
    vsc = ClassifierModelFull(args.dataset, width, height, channels, 
                                  args.kernel_size, args.hidden_size, args.latent_size, 
                                  args.lr, args.alpha, device, args.log_interval,
                                  args.normalize,
                                  encoder_model_path= args.encoder_model_path,
                                  model_type= f"cvscc_sparse_{args.fold_number}" if not args.use_vae_encoder else f"cvscc_variational_{args.fold_number}",
                                  encoder_model= "Sparse_Encoder" if not args.use_vae_encoder else "Vae_Encoder",
                                  flatten=False,
                                  train_encoder=args.train_encoder,
                                  )
    vsc.run_training(train_loader, test_loader, args.epochs,
                     args.report_interval, args.sample_size, 
                     reload_model=not args.do_not_resume, print_acc=True)
    