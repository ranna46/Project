# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 14:36:08 2020

@author: Ranna
"""

import pandas as pd
import numpy as np
import statistics
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pickle
import requests
import json

df=pd.read_csv("C:\\Users\\Ranna\\Documents\\excelR\\Project\\data.csv", encoding='latin1')

df.drop(df.columns[[0,3,5,6,7,9,11,12,13,23,30,31,32]],axis=1,inplace= True)

df.rename(columns={"loan_amnt ":"loan_amnt"},inplace=True)
df.rename(columns={"total revol_bal":"total_revol_bal"},inplace=True)

df['loan_amnt'] = df['loan_amnt'].astype(float)
df['terms'] = df['terms'].astype(str).str.rstrip(' months')
df['terms'] = df['terms'].astype(float)
df['Experience'] = df['Experience'].astype(str).str.lstrip('<').str.rstrip('+ years')
df['Experience'] = df['Experience'].astype(float)


###########################################################################################
df.Experience = df.Experience.fillna(df.Experience.median())

# filling with mean
df.annual_inc = df.annual_inc.fillna(df.annual_inc.mean())
df.delinq_2yrs = df.delinq_2yrs.fillna(df.delinq_2yrs.mean())
df.inq_last_6mths = df.inq_last_6mths.fillna(df.inq_last_6mths.mean())
df.numb_credit = df.numb_credit.fillna(df.numb_credit.mean())
df.total_credits = df.total_credits.fillna(df.total_credits.mean())
df.pub_rec = df.pub_rec.fillna(df.pub_rec.mean())
df.collections_12_mths_ex_med = df.collections_12_mths_ex_med.fillna(df.collections_12_mths_ex_med.mean())
df.tot_colle_amt = df.tot_colle_amt.fillna(df.tot_colle_amt.mean())
df.tot_curr_bal = df.tot_curr_bal.fillna(df.tot_curr_bal.mean())
df.acc_now_delinq = df.acc_now_delinq.fillna(df.acc_now_delinq.mean())

# filling with 0, assuming that maybe no dilinquency was committed and no public record were made and no derogatories were filed
df.mths_since_last_delinq = df.mths_since_last_delinq.fillna(0)
df.mths_since_last_record = df.mths_since_last_record.fillna(0)
df.mths_since_last_major_derog = df.mths_since_last_major_derog.fillna(0)
###################################################################################################

### Splitting the data into train and test data 
#df.values.reshape(-1,1)
from sklearn.model_selection import train_test_split
df_train,df_test  = train_test_split(df,test_size = 0.2, random_state = 123)

regressor = smf.ols("np.sqrt(total_revol_bal)~ np.sqrt(tot_curr_bal)+(tot_curr_bal*tot_curr_bal)+ loan_amnt+(loan_amnt*loan_amnt) + np.sqrt(annual_inc) +(annual_inc*annual_inc)+ np.sqrt(numb_credit)+(numb_credit*numb_credit) + np.sqrt(total_credits)+(total_credits*total_credits) + np.sqrt(total_rec_int)+(total_rec_int*total_rec_int) + np.sqrt(mths_since_last_record)+(mths_since_last_record*mths_since_last_record)+np.sqrt(pub_rec)+np.sqrt(terms)+(terms*terms)+np.sqrt(mths_since_last_major_derog)+(mths_since_last_major_derog*mths_since_last_major_derog)+np.sqrt(Experience)+(Experience*Experience)+np.sqrt(debt_income_ratio)+(debt_income_ratio*debt_income_ratio)+np.sqrt(mths_since_last_delinq)+(mths_since_last_delinq*mths_since_last_delinq)+np.sqrt(Rate_of_intrst)+(Rate_of_intrst*Rate_of_intrst)+np.sqrt(delinq_2yrs)+(delinq_2yrs*delinq_2yrs)+np.sqrt(collections_12_mths_ex_med)+np.sqrt(tot_colle_amt)+(tot_colle_amt*tot_colle_amt)+np.sqrt(total_rec_late_fee)+np.sqrt(acc_now_delinq)", data=df_train).fit()
#regressor.summary()# 0.44

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[5,2,5,4,8,6,8,5,5,5,8,4,6,8,6,5,7,6,2]]))



