import pandas as pd
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
import numpy as np

train = pd.read_csv('../data/train.csv')

'''
vals = []
for i in np.arange(train.shape[0]):
    vals.append(str(train.num_var5.iloc[i]) + '_' + str(train.num_var5_0.iloc[i])) 
res = pd.factorize(vals)
print res
train['num_var5_comb'] = res[0]
print train['num_var5_comb']

sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(plt.hist, "num_var5_comb") \
   .add_legend()
plt.title('num_var5_comb')
plt.show()
'''


vals = []
for i in np.arange(train.shape[0]):
    vals.append(str(train.num_var12.iloc[i]) + '_' + str(train.num_var12_0.iloc[i])) 
res = pd.factorize(vals)
print res
train['num_var12_comb'] = res[0]
#print train['num_var6_comb']

sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(plt.hist, "num_var12") \
   .add_legend()
plt.title('num_var12')
plt.show()

sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(plt.hist, "num_var12_0") \
   .add_legend()
plt.title('num_var12_0')
plt.show()

sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(plt.hist, "num_var12_comb") \
   .add_legend()
plt.title('num_var12_comb')
plt.show()

