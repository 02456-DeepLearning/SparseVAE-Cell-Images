import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pdb
import numpy as np

# Load all data
df = pd.read_csv("/zhome/0f/f/137116/Desktop/DeepLearningProject/SparseVAE-Cell-Images/bbbc021/singlecell/metadata.csv")

# Balance dataset
label='moa'
g = df.groupby(label, group_keys=False)
balanced_df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()))).reset_index(drop=True)
df = balanced_df

# Generate 5 stratified folds containing indexes
skf = StratifiedKFold(n_splits=5)
target = df.loc[:,'moa']

fold_no = 1
for train_index, val_index in skf.split(df, target):
  # Extract indexes corresponding to original dataframe 
  original_train_idx = np.array([df.iloc[train_index[i]][0] for i in range(len(train_index))])
  original_test_idx = np.array([df.iloc[val_index[i]][0] for i in range(len(val_index))]
                               )
  train_fold_df = pd.DataFrame(original_train_idx)
  test_fold_df = pd.DataFrame(original_test_idx)

  train_fold_df.to_csv('/zhome/0f/f/137116/Desktop/DeepLearningProject/SparseVAE-Cell-Images/datasplit/' + 'train_fold_' + str(fold_no) + '.csv')
  test_fold_df.to_csv('/zhome/0f/f/137116/Desktop/DeepLearningProject/SparseVAE-Cell-Images/datasplit/' + 'test_fold_' + str(fold_no) + '.csv')

  fold_no += 1
  
  