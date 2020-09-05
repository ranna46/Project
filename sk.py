# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 01:17:45 2020

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
df.values.reshape(-1,1)
X = df[['loan_amnt', 'terms', 'Rate_of_intrst', 'Experience', 'annual_inc','debt_income_ratio', 'delinq_2yrs', 
       'mths_since_last_delinq', 'mths_since_last_record', 'numb_credit','pub_rec', 'total_credits', 'total_rec_int',
       'total_rec_late_fee', 'collections_12_mths_ex_med', 'mths_since_last_major_derog','acc_now_delinq', 'tot_colle_amt', 
       'tot_curr_bal']]
y = df[['total_revol_bal']]

X["loan_sqrd"] = X.loan_amnt*X.loan_amnt
X["loan_sqrt"] = np.sqrt(X.loan_amnt)

X["terms_sqrd"] = X.terms*X.terms
X["terms_sqrt"] = np.sqrt(X.terms)

X["Rate_of_intrst_sqrd"] = X.Rate_of_intrst*X.Rate_of_intrst
X["Rate_of_intrst_sqrt"] = np.sqrt(X.Rate_of_intrst)

X["Rate_of_intrst_sqrd"] = X.Rate_of_intrst*X.Rate_of_intrst
X["Rate_of_intrst_sqrt"] = np.sqrt(X.Rate_of_intrst)


X["Experience_sqrd"] = X.Experience*X.Experience
X["Experience_sqrt"] = np.sqrt(X.Experience)


X["annual_inc_sqrd"] = X.annual_inc*X.annual_inc
X["annual_inc_sqrt"] = np.sqrt(X.annual_inc)


X["debt_income_ratio_sqrd"] = X.debt_income_ratio*X.debt_income_ratio
X["debt_income_ratio_sqrt"] = np.sqrt(X.debt_income_ratio)

X["delinq_2yrs_sqrd"] = X.delinq_2yrs*X.delinq_2yrs
X["delinq_2yrs_sqrt"] = np.sqrt(X.delinq_2yrs)


X["mths_since_last_delinq_sqrd"] = X.mths_since_last_delinq*X.mths_since_last_delinq
X["mths_since_last_delinq_sqrt"] = np.sqrt(X.mths_since_last_delinq)


X["mths_since_last_record_sqrd"] = X.mths_since_last_record*X.mths_since_last_record
X["mths_since_last_record_sqrt"] = np.sqrt(X.mths_since_last_record)


X["numb_credit_sqrd"] = X.numb_credit*X.numb_credit
X["numb_credit_sqrt"] = np.sqrt(X.numb_credit)

X["pub_rec_sqrd"] = X.pub_rec*X.pub_rec
X["pub_rec_sqrt"] = np.sqrt(X.pub_rec)


X["total_credits_sqrd"] = X.total_credits*X.total_credits
X["total_credits_sqrt"] = np.sqrt(X.total_credits)

X["total_rec_int_sqrd"] = X.total_rec_int*X.total_rec_int
X["total_rec_int_sqrt"] = np.sqrt(X.total_rec_int)

#X["total_rec_late_fee_sqrd"] = X.total_rec_late_fee*X.total_rec_late_fee
X["total_rec_late_fee_sqrt"] = np.sqrt(X.total_rec_late_fee)

X["collections_12_mths_ex_med_sqrd"] = X.collections_12_mths_ex_med*X.collections_12_mths_ex_med
X["collections_12_mths_ex_med_sqrt"] = np.sqrt(X.collections_12_mths_ex_med)


X["mths_since_last_major_derog_sqrd"] = X.mths_since_last_major_derog*X.mths_since_last_major_derog
X["mths_since_last_major_derog_sqrt"] = np.sqrt(X.mths_since_last_major_derog)


#X["acc_now_delinq_sqrd"] = X.acc_now_delinq*X.acc_now_delinq
X["acc_now_delinq_sqrt"] = np.sqrt(X.acc_now_delinq)

X["tot_colle_amt_sqrd"] = X.tot_colle_amt*X.tot_colle_amt
X["tot_colle_amt_sqrt"] = np.sqrt(X.tot_colle_amt)

X["tot_curr_bal_sqrd"] = X.tot_curr_bal*X.tot_curr_bal
X["tot_curr_bal_sqrt"] = np.sqrt(X.tot_curr_bal)


y["total_revol_bal_sqrt"] = np.sqrt(y.total_revol_bal)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model2 = LinearRegression()
model2.fit(X_train, y_train.total_revol_bal_sqrt)
pred2 = model2.predict(X_train)
pred2n=pred2*pred2
model2.score(X_train, y_train.total_revol_bal_sqrt)
rmse2 = np.sqrt(np.mean((pred2n-y_train.total_revol_bal_sqrt)**2))
rmse2

pred3 = model2.predict(X_test)
pred3n=pred3*pred3
model2.score(X_test, y_test.total_revol_bal_sqrt)
rmse3 = np.sqrt(np.mean((pred3n-y_test.total_revol_bal_sqrt)**2))
rmse3









