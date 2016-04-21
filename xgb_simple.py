import pandas as pd
import xgboost as xgb

def get_params():
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.5
    params["min_child_weight"] = 1
    params["subsample"] = 0.9
    params["colsample_bytree"] = 0.9
    params["silent"] = 1
    params["max_depth"] = 10
    plst = list(params.items())
    return plst


xgb_num_rounds = 10

train = pd.read_csv("data/train.csv") # the train dataset is now a Pandas DataFrame
test = pd.read_csv("data/test.csv") # the train dataset is now a Pandas DataFrame

train_columns_to_drop = ['ID', 'TARGET']
test_columns_to_drop = ['ID']

train_feat = train.drop(train_columns_to_drop, axis=1)
test_feat = test.drop(test_columns_to_drop, axis=1)


xgtrain = xgb.DMatrix(train_feat, train.TARGET.values)
xgtest = xgb.DMatrix(test_feat)

plst = get_params()
print(plst)
model = xgb.train(plst, xgtrain, xgb_num_rounds)
test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)


preds_out = pd.DataFrame({"ID": test['ID'].values, "TARGET": test_preds})
preds_out = preds_out.set_index('ID')
preds_out.to_csv('xgb_simple.csv')
print 'finish'

