import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/zhome/bb/9/153142/SparseVAE-Cell-Images/bbbc021/singlecell/metadata.csv')
nuniq = len(np.unique(data['Image_Metadata_Compound']))

print("\n")
freq_compound = data['Image_Metadata_Compound'].value_counts()
print("Compound Frequency")
print(freq_compound)
print("\n")

print("MOA Frequency")
freq_moa = data['moa'].value_counts()
print(freq_moa)

freq_compound.plot.bar()
plt.show()
freq_moa.plot.bar()
plt.show()
