import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import StratifiedKFold

def get_params():
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.5
    params["eval_meric"] = 'auc'
    params["min_child_weight"] = 1
    params["subsample"] = 0.9
    params["colsample_bytree"] = 0.9
    params["silent"] = 1
    params["max_depth"] = 10
    plst = list(params.items())
    return plst

if __name__ == '__main__':
        
	#xgb_num_rounds = 100
	xgb_num_rounds = 10

	train = pd.read_csv("data/train.csv") # the train dataset is now a Pandas DataFrame
	test = pd.read_csv("data/test.csv") # the train dataset is now a Pandas DataFrame

	train_columns_to_drop = ['ID', 'TARGET']
	test_columns_to_drop = ['ID']
	id_target_col = ['ID', 'TARGET']

	#train_feat = train.drop(train_columns_to_drop, axis=1)
	#test_feat = test.drop(test_columns_to_drop, axis=1)

	params = get_params()
	print(params)
	split = 5
	skf = StratifiedKFold(train.TARGET.values,
                          n_folds=split,
                          shuffle=False,
                          random_state=42)

	for train_index, valid_index in skf:
		train_cv = train.iloc[train_index]
		valid_cv = train.iloc[valid_index]

		X_train_cv = train_cv.drop(id_target_col, axis=1)
		y_train_cv = train_cv.TARGET.values
		X_valid_cv = valid_cv.drop(id_target_col, axis=1)
		y_valid_cv = valid_cv.TARGET.values

		xgtrain_cv = \
			xgb.DMatrix(csr_matrix(X_train_cv),
						y_train_cv,silent=False)

		xgvalid_cv = \
			xgb.DMatrix(csr_matrix(X_valid_cv),
						y_valid_cv,silent=False)
		
		watchlist = [(xgvalid_cv, 'eval'), (xgtrain_cv, 'train')]
		clf = xgb.train(params, xgtrain_cv, xgb_num_rounds,
                        evals=watchlist, early_stopping_rounds=50,
                        verbose_eval=True)

		valid_preds = clf.predict(xgvalid_cv)
        print('Valid Log Loss:', log_loss(y_valid_cv,
                                          valid_preds))
        print('Valid ROC:', roc_auc_score(y_valid_cv,
                                          valid_preds))
						


	'''
	xgtrain = xgb.DMatrix(train_feat, train.TARGET.values)
	xgtest = xgb.DMatrix(test_feat)

	model = xgb.train(plst, xgtrain, xgb_num_rounds)
	test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)


	preds_out = pd.DataFrame({"ID": test['ID'].values, "TARGET": test_preds})
	preds_out = preds_out.set_index('ID')
	preds_out.to_csv('xgb_simple.csv')
	print 'finish'
	'''


