import pandas as pd
import numpy as np

train = pd.read_csv('../data/train.csv')

num_cols = filter(lambda x: x.startswith('num'), train.columns)

for col in num_cols:
    uni_val = np.unique(train[col])
    print '%s\t%s' %(col, str(uni_val))
    dtype = train[col].dtype

