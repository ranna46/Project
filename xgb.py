#importing necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pickle
## reading the dataset
df=pd.read_csv("C:\\Users\\Ranna\\Documents\\excelR\\Project\\data.csv", encoding='latin1')
df

## dropping unnecessary columns
df.drop(df.columns[[0,3,5,6,7,9,11,12,13,23,30,31,32]],axis=1,inplace= True)

## renaming a few columns for better readability
df.rename(columns={"loan_amnt ":"loan_amnt"},inplace=True)
df.rename(columns={"total revol_bal":"total_revol_bal"},inplace=True)

## converting data type
df['loan_amnt'] = df['loan_amnt'].astype(float)
df['terms'] = df['terms'].astype(str).str.rstrip(' months')
df['terms'] = df['terms'].astype(float)
df['Experience'] = df['Experience'].astype(str).str.lstrip('<').str.rstrip('+ years')
df['Experience'] = df['Experience'].astype(float)

#df.values.reshape(-1,1)
X = df[['total_credits', 'collections_12_mths_ex_med', 'terms', 'annual_inc', 'numb_credit', 'delinq_2yrs', 'Experience', 'pub_rec', 'acc_now_delinq','debt_income_ratio', 'Rate_of_intrst', 'mths_since_last_major_derog', 'tot_colle_amt', 'tot_curr_bal', 'loan_amnt', 'total_rec_late_fee', 'mths_since_last_record', 'mths_since_last_delinq', 'total_rec_int']]
y = df[['total_revol_bal']]
#total_credits, collections_12_mths_ex_med, terms, annual_inc, numb_credit, delinq_2yrs, Experience, pub_rec, acc_now_delinq, debt_income_ratio, Rate_of_intrst, mths_since_last_major_derog, tot_colle_amt, tot_curr_bal, loan_amnt, total_rec_late_fee, mths_since_last_record, mths_since_last_delinq, total_rec_int

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, explained_variance_score, r2_score
from xgboost.sklearn import XGBClassifier

#data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#dtrain = xgb.DMatrix(X_train, label=y_train)
#dtest = xgb.DMatrix(X_test, label=y_test)

regressor = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.4,
                max_depth = 6, alpha = 15, n_estimators = 20,random_state=123)

regressor.fit(X_train.as_matrix(),y_train.as_matrix())

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(X_train.as_matrix()))


#'5','2','5','4','8','6','8','5','5','5','8','4','6','8','6','5','7','6','2'
#{'tot_curr_bal','loan_amnt','terms', 'Rate_of_intrst', 'Experience', 'annual_inc', 'debt_income_ratio', 'delinq_2yrs', 'mths_since_last_delinq', 'mths_since_last_record', 'numb_credit','pub_rec', 'total_credits', 'total_rec_int','total_rec_late_fee', 'collections_12_mths_ex_med', 'mths_since_last_major_derog','acc_now_delinq', 'tot_colle_amt'}, 






