import numpy as np

def remove_constant_and_duplicate(df):
    colsToRemove = []
    for col in df.columns:
        if df[col].std() == 0:
            colsToRemove.append(col)
    df.drop(colsToRemove, axis=1, inplace=True)

    colsToRemove = []
    columns = df.columns
    for i in range(len(columns)-1):
        v = df[columns[i]].values
        for j in range(i+1,len(columns)):
            if np.array_equal(v,df[columns[j]].values):
                colsToRemove.append(columns[j])
    df.drop(colsToRemove, axis=1, inplace=True)
