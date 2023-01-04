import pdb 
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
import glob
import torch

class CelebA(Dataset):
    def __init__(self, dataset_path, train=True, download=False,
                 transform=None, random_state=42, train_size=100_000, 
                 test_size=20_000):
        self.random_state = random_state
        self.dataset_path = os.path.join(dataset_path)
        self.image_files = np.array(os.listdir(self.dataset_path))
        self.train_size = train_size
        self.test_size = test_size
        self.transform = transform
        
        np.random.seed(random_state)
        np.random.shuffle(self.image_files)

        if train:
            self.image_files = self.image_files[:train_size]
        else:
            self.image_files = self.image_files[-test_size:]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_path,
                                self.image_files[idx])
        image, target = io.imread(img_name), 0

        if self.transform:
            image = self.transform(image)

        return image, target
    
    
class DSprites(Dataset):
    def __init__(self, dataset_path, train=True, download=False,
                 transform=None, random_state=42, train_size=600_000,
                 test_size=100_000):
        self.random_state = random_state
        self.dataset_path = dataset_path
        images = np.load(os.path.join(dataset_path,
                'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'))['imgs']
        self.images = images[..., None] * 255
        self.train_size = train_size
        self.test_size = test_size
        self.transform = transform
        
        np.random.seed(random_state)
        np.random.shuffle(self.images)
        
        if train:
            self.images = self.images[:train_size]
        else:
            self.images = self.images[-test_size:]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image, target = self.images[idx], 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


class Cell(Dataset):
    def __init__(self, dataset_path, train=True, download=False,
                 transform=None, random_state=42, train_size=.8,
                 test_size=-1, fold=None):

            
        self.random_state = random_state
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.test_size = test_size

        


        self.transform = transform
        self.fold = fold
        
        # pandas uses relative path for some reason :D
        self.df = pd.read_csv("./bbbc021/singlecell/metadata.csv")


        label='moa'

        g = self.df.groupby(label, group_keys=False)
        balanced_df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()))).reset_index(drop=True)
        self.df = balanced_df

        self.image_paths = []

        base_path = "./bbbc021/singlecell/singh_cp_pipeline_singlecell_images/"
        helper_fn = lambda x,y: base_path+x+'/'+y
        
        self.image_paths = [helper_fn(x,y) for x,y in zip(self.df['Multi_Cell_Image_Name'], self.df['Single_Cell_Image_Name'])]
        
        self.targets = self.df['moa']
        self.targets = pd.factorize(self.targets)[0]
        

        #print(np.load(self.image_paths[0]))
        # pdb.set_trace()
        np.random.seed(random_state)

        # shuffle image_paths and targets in unison
        new_idxs = np.random.permutation(len(self.image_paths))

        self.image_paths = np.array(self.image_paths)[new_idxs]
        self.targets = np.array(self.targets)[new_idxs]

        # for i in range(len(df['Multi_Cell_Image_Name'])):
        #     self.image_paths.append("SparseVAE-Cell-Images/bbbc021/singlecell/singh_cp_pipeline_singlecell_images/" + df['Multi_Cell_Image_Name'][i] + "/"+ df['Single_Cell_Image_Name'][i])

        #  = glob.glob("SparseVAE-Cell-Images/bbbc021/singlecell/singh_cp_pipeline_singlecell_images/**/*.npy", recursive=True)
    
        data_size = len(self.image_paths)
        if train:
            self.image_paths = self.image_paths[:int(train_size*data_size)]
            self.targets = self.targets[:int(train_size*data_size)]
        else:
            self.image_paths = self.image_paths[-int((1-train_size)*data_size):]
            self.targets = self.targets[-int((1-train_size)*data_size):]
        
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image, target = self.image_paths[idx], self.targets[idx] # 0 -> not implemented yet
        

        image = np.load(image)
        
        #image = image.reshape(-1)
        #image = image[:784].reshape([64,64])

        if self.transform:
            image = image.astype(np.float)

            # image dimensions = (68, 68, 3)
            image[:,:,0] /= np.max(image[:,:,0])
            image[:,:,1] /= np.max(image[:,:,1])
            image[:,:,2] /= np.max(image[:,:,2])

            
            image= np.transpose(image, (0, 1, 2)) # reshaped to channel, x, y

            # todo add augmentation
            image = self.transform(image.astype(np.float))

            

            # image = image.to(float)
            # #print(type(image[0,0,0]))
            image = image.type(torch.FloatTensor)
        
        return image, target
        
        
class StratifiedCell(Dataset):
    def __init__(self, dataset_path="./bbbc021/singlecell/metadata.csv", train=True, download=False,
                 transform=None, random_state=42, fold = 1):
    
        self.random_state = random_state
        self.transform = transform
        self.fold = fold - 1
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path)
            
        # Load training data indexes
        self.trainfold1 = pd.read_csv("./datasplit/train_fold_1.csv")
        self.trainfold2 = pd.read_csv("./datasplit/train_fold_2.csv")
        self.trainfold3 = pd.read_csv("./datasplit/train_fold_3.csv")
        self.trainfold4 = pd.read_csv("./datasplit/train_fold_4.csv")
        self.trainfold5 = pd.read_csv("./datasplit/train_fold_5.csv")

        # Load test data indexes
        self.testfold1 = pd.read_csv("./datasplit/test_fold_1.csv")
        self.testfold2 = pd.read_csv("./datasplit/test_fold_2.csv")
        self.testfold3 = pd.read_csv("./datasplit/test_fold_3.csv")
        self.testfold4 = pd.read_csv("./datasplit/test_fold_4.csv")
        self.testfold5 = pd.read_csv("./datasplit/test_fold_5.csv")
        
        self.train_folds = [self.trainfold1,self.trainfold2,self.trainfold3,self.trainfold4,self.trainfold5]
        self.test_folds = [self.testfold1,self.testfold2,self.testfold3,self.testfold4,self.testfold5]

        # Create correct image paths for all images
        self.image_paths = []
        base_path = "./bbbc021/singlecell/singh_cp_pipeline_singlecell_images/"
        helper_fn = lambda x,y: base_path+x+'/'+y
        
        self.image_paths = [helper_fn(x,y) for x,y in zip(self.df['Multi_Cell_Image_Name'], self.df['Single_Cell_Image_Name'])]
        self.image_paths = np.array(self.image_paths)
        self.targets = self.df['moa']
        self.targets, labels = pd.factorize(self.targets)
        
    
        if train:
            self.image_paths = list(self.image_paths[np.array(self.train_folds[self.fold].iloc[:,1])])
            self.targets = list(self.targets[np.array(self.train_folds[self.fold].iloc[:,1])])
        else:
            self.image_paths = list(self.image_paths[np.array(self.test_folds[self.fold].iloc[:,1])])
            self.targets = list(self.targets[np.array(self.test_folds[self.fold].iloc[:,1])])

        print('StratifiedCell image_path size', len(self.image_paths))
        print('StratifiedCell targets size', len(self.targets))

        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image, target = self.image_paths[idx], self.targets[idx] # 0 -> not implemented yet
        

        image = np.load(image)
        
        #image = image.reshape(-1)
        #image = image[:784].reshape([64,64])

        if self.transform:
            image = image.astype(np.float)

            # image dimensions = (68, 68, 3)
            image[:,:,0] /= np.max(image[:,:,0])
            image[:,:,1] /= np.max(image[:,:,1])
            image[:,:,2] /= np.max(image[:,:,2])

            
            image= np.transpose(image, (0, 1, 2)) # reshaped to channel, x, y

            # todo add augmentation
            
            image = self.transform(image.astype(np.float))

            

            # image = image.to(float)
            # #print(type(image[0,0,0]))
            image = image.type(torch.FloatTensor)
        
        return image, target