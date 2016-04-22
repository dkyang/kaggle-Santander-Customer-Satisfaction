import pandas as pd
from preprocess import remove_constant_and_duplicate
from util import calc_MI_feat_target

train = pd.read_csv('data/train.csv')
remove_constant_and_duplicate(train)

for col in train.columns:
    mi = calc_MI_feat_target(train[col], train.TARGET.values, 20)
    print mi
