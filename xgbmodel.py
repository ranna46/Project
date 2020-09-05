# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:01:44 2020

@author: Ranna
"""
#importing necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import statsmodels.formula.api as smf
import statsmodels.api as sm
#get_ipython().run_line_magic('matplotlib', 'inline')

## reading the dataset
df=pd.read_csv("C:\\Users\\Ranna\\Documents\\excelR\\Project\\data.csv", encoding='latin1')
df

## dropping unnecessary columns
df.drop(df.columns[[0,3,5,6,7,9,11,12,13,23,30,31,32]],axis=1,inplace= True)
df.shape
df.info()

## renaming a few columns for better readability
df.rename(columns={"loan_amnt ":"loan_amnt"},inplace=True)
df.rename(columns={"total revol_bal":"total_revol_bal"},inplace=True)

## converting data type
df['loan_amnt'] = df['loan_amnt'].astype(float)

df['terms'] = df['terms'].astype(str).str.rstrip(' months')
df['terms'] = df['terms'].astype(float)
df.terms

df['Experience'] = df['Experience'].astype(str).str.lstrip('<').str.rstrip('+ years')
df['Experience'] = df['Experience'].astype(float)
df.Experience


X = df[['loan_amnt', 'terms', 'Rate_of_intrst', 'Experience', 'annual_inc','debt_income_ratio', 'delinq_2yrs', 
       'mths_since_last_delinq', 'mths_since_last_record', 'numb_credit','pub_rec', 'total_credits', 'total_rec_int',
       'total_rec_late_fee', 'collections_12_mths_ex_med', 'mths_since_last_major_derog','acc_now_delinq', 'tot_colle_amt', 
       'tot_curr_bal']]
y = df[['total_revol_bal']]


###############################################################
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, explained_variance_score, r2_score
from xgboost.sklearn import XGBClassifier
from xgboost import cv

data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)



###########
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.4,
                max_depth = 6, alpha = 15, n_estimators = 20,random_state=123)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#17795.751
print(explained_variance_score(preds,y_test))
xg_reg.score(X_train,y_train)#0.44
xg_reg.score(X_test,y_test)#0.32
############

################
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.5,
                max_depth = 8, gamma=0.2, n_estimators = 20,random_state=2, seed = 10, min_child_weight=5)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#17719.365

xg_reg.score(X_train,y_train)#0.48
xg_reg.score(X_test,y_test)#0.32
###################
xg_reg = xgb.XGBRegressor(objective ='reg:linear',  learning_rate = 0.5,
                max_depth = 5,  n_estimators = 25,random_state=123)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#18133.21

xg_reg.score(X_train,y_train)#0.444
xg_reg.score(X_test,y_test)#0.38
################
xg_reg = xgb.XGBRegressor(objective ='reg:linear')

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#18648.523

xg_reg.score(X_train,y_train)#0.45
xg_reg.score(X_test,y_test)#0.345
#########################
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.7,
                max_depth = 5, alpha = 10, n_estimators = 50,random_state=1234, seed = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#17842.30

xg_reg.score(X_train,y_train)#0.49
xg_reg.score(X_test,y_test)#0.31
########
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.4, learning_rate = 0.7,
                max_depth = 5, alpha = 10, n_estimators = 50,random_state=1234, seed = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#17822.358

xg_reg.score(X_train,y_train)#0.50
xg_reg.score(X_test,y_test)#0.32
###############
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.6, #####
                max_depth = 5, alpha = 10, n_estimators = 50,random_state=1234, seed = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#17474.94

xg_reg.score(X_train,y_train)#0.48
xg_reg.score(X_test,y_test)#0.34
##############
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.6,
                max_depth = 10, alpha = 10, n_estimators = 50,random_state=1234, seed = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#18370.864

xg_reg.score(X_train,y_train)#0.67
xg_reg.score(X_test,y_test)#0.27
###############
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.6,
                max_depth = 5, alpha = 15, n_estimators = 50,random_state=1234, seed = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#18082.965

xg_reg.score(X_train,y_train)#0.47
xg_reg.score(X_test,y_test)#0.38
#############
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.5,
                max_depth = 5, alpha = 20, n_estimators = 50,random_state=1234, seed = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#17719.36

xg_reg.score(X_train,y_train)#0.48
xg_reg.score(X_test,y_test)#0.32
################
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.76,
                max_depth = 6, gamma = 5.14, n_estimators = 10,reg_alpha= 79.0, reg_lambda=0.43)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#18023.303

xg_reg.score(X_train,y_train)#0.50
xg_reg.score(X_test,y_test)#0.38
#################
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.5,
                max_depth = 5, alpha = 25, n_estimators = 50,random_state=1234, seed = 1000)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#17968.39

xg_reg.score(X_train,y_train)#0.48
xg_reg.score(X_test,y_test)#0.39
###########
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.8, learning_rate = 0.1, scale_pos_weight=1, nthread=4,
                max_depth = 5, min_child_weight=1,  gamma=0, subsample=0.8, alpha = 15, n_estimators = 1000, seed = 27)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#18022.281

xg_reg.score(X_train,y_train)#0.62
xg_reg.score(X_test,y_test)#0.33
##
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27,silent = 1), 
 param_grid = param_test1, scoring='r2',n_jobs=4,iid=False, cv=5)

gsearch1.fit(X_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
##############################


xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9, 'max_depth': 6,'subsample': 0.9, 'lambda': 1.,
            'nthread': -1, 'booster' : 'gbtree', 'silent': 1,'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, data_dmatrix, verbose_eval=1)
predict = model.predict(data_dmatrix)
model.score(X_train,y_train)#0.62
xg_reg.score(X_test,y_test)#0.33
############

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
watchlist = [(dtrain, 'train'), (dtest, 'test')]
xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9, 
        'max_depth': 6, 'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 10, watchlist, early_stopping_rounds=2, maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)
####################
gbm_param_grid = {
     'colsample_bytree': np.linspace(0.5, 0.9, 5),
     'n_estimators':[100, 200],
     'max_depth': [10, 15, 20, 25]
}
gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator = gbm, param_grid = gbm_param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose = 1)

grid_mse.fit(X_train, y_train)
scores = cross_val_score(grid_mse, X, y, scoring='r2', cv=5) 

print("Best parameters found: ",grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))

pred = grid_mse.predict(X_test)

print("Root mean square error for test dataset: {}".format(np.round(np.sqrt(mean_squared_error(y_test, pred)), 2)))


#########################################################################

# declare parameters
params = {
            'objective':'reg:linear',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':20
         }         
# instantiate the classifier 
xgb_clf = xgb.XGBRegressor(**params)

# fit the classifier to the training data
xgb_clf.fit(X_train, y_train)
# we can view the parameters of the xgb trained model as follows -

print(xgb_clf)
# make predictions on test data

y_pred = xgb_clf.predict(X_test)

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)

xgb_cv.head()

xgb.plot_importance(xgb_clf)
plt.figure(figsize = (16, 12))
plt.show()

#pip install hyperopt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'learning_rate' : hp.uniform('learning_rate', 0, 2),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.uniform('n_estimators', 5, 200),
        'seed': 2
    }

def objective(space):
    clf=xgb.XGBRegressor(
                    n_estimators =space['n_estimators'], max_depth = float(space['max_depth']), gamma = space['gamma'],
                    reg_lambda = float(space['reg_lambda']),min_child_weight=float(space['min_child_weight']),
                    colsample_bytree=float(space['colsample_bytree']), learning_rate=float(space['learning_rate']))
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="rmse",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X_test)
    Rsq = r2_score(y_test, pred)
    print ("SCORE:", Rsq)
    return {'loss': -Rsq, 'status': STATUS_OK }

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 10,
                        trials = trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)

##################Final model#################
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.6, 
                max_depth = 5, alpha = 10, n_estimators = 50,random_state=1234, seed = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
prediction = xg_reg.predict(X)


rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))#17474.98

xg_reg.score(X_train,y_train)#0.48
xg_reg.score(X_test,y_test)#0.34
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.6,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

cv_results.head()
print((cv_results["test-rmse-mean"]).tail(1))

#scores = cross_val_score(xg_reg, X_test, y_test, scoring='r2', cv=3) 
#scores
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


df['prediction'] = prediction
Result = pd.DataFrame(df, columns = ['total_revol_bal', 'prediction']) 
Result
####################################################################################











