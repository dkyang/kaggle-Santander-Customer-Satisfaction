import pandas as pd

train = pd.read_csv('../data/train.csv')

print train.shape
print train.columns

for col in train.columns:
    print '%s %s' % (col, train[col].dtype)


test = pd.read_csv('../data/test.csv')
print test.shape
