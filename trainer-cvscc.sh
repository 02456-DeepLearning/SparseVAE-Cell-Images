#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J cvscc_fold1
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# request 32GB of system-memory
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
#BSUB -u s204159@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o cvscc_fold1-%J.out
#BSUB -e cvscc_fold1-%J.err
# -- end of LSF options --

nvidia-smi
### Load the cuda module
module load python3/3.7.14
module load cuda/10.2
module load cudnn/v8.3.2.44-prod-cuda-10.2
module load ffmpeg/4.2.2

### cd /zhome/bb/9/153142
### source GAN-env/bin/activate
### cd /zhome/bb/9/153142/GAN-env/esrgan-tf2
### python train_psnr.py
### python train_esrgan.py
source /zhome/a2/4/155672/Desktop/DeepLearning/SparseVAE-Cell-Images/dl-env/bin/activate
python3 /zhome/a2/4/155672/Desktop/DeepLearning/SparseVAE-Cell-Images/src/train-cvscc.py --fold-number=1 --encoder-model-path "./results/checkpoints/convvae_1_ConvVAE_stratifiedcell_1_400_202_0-001_400.pth" --epochs 101  --report-interval 20  --dataset stratifiedcell  --lr 0.001 --latent-size 30 --hidden-size 60 --do-not-resume


