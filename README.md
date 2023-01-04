# GENERATIVE MODELLING FOR PHENOTYPIC PROFILING

## Description 
The following git repo contains the project code for 02456 Deep learning 2022. The code is originally from  [Variational-Sparse-Coding](https://github.com/Alfo5123/Variational-Sparse-Coding). The code as been modified for use in the course. To see changes from original code repository look at git commits. The code has been altered by the following authors.

## Authors

 - Niels Raunkjær Holm (s204503)
 - Christian Kento Ramussen (s204159))
 - Mathias Frosz Nielsen  (s201968)
 - Asbjørn Ebenezer Magnussen  (s183546)

## Usage

### Setup

The code requires the following python version and the following modules (Can be loaded with `module load` command on the HPC at DTU)

```
python3/3.7.14
cuda/10.2
cudnn/v8.3.2.44-prod-cuda-10.2
ffmpeg/4.2.2
```

The following lines will clone the repository and install all the required dependencies.

```
$ https://github.com/02456-DeepLearning/SparseVAE-Cell-Images.git
$ cd Variational-Sparse-Coding
$ pip install -r requirements.txt
$ pip install pillow
$ pip install tensorflow==2.2.0
$ pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
$ pip install protobuf==3.20.
```

### Datasets

To access the preprocessed dataset please contact Bjørn Sand Jensen at DTU Compute.

### Pretrained Models

### Train Models 

The final trained models are located in results_shared.


To train the convolutional variational autoencoder call the following command (remember to change folder paths in code):

```
$ python3 src/train-convvae.py --dataset cell --epochs 75 --report-interval 5 --lr 0.001 --do-not-resume --latent-size 200 --hidden-size 400 --fold-number=1
```

To train the sparse convolutional variational autoencoder call the following command
```
$ python3 src/train-convvsc.py --dataset cell --epochs 80 --report-interval 5 --lr 0.001 --latent-size 200 --hidden-size 400 --beta-delta 0.0001 --fold-number=1
```

To visualize training results in TensorBoard, we can use the following command from a new terminal within **src** folder. 

```
$ tensorboard --logdir='./logs' --port=6006
```

To visualise the effect of latent dimensions use the following command

```
python3 src/sliders.py
```


## License
[MIT License](https://github.com/Alfo5123/Variational-Sparse-Coding/blob/master/LICENSE)

