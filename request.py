# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:21:56 2020

@author: Ranna
"""

import requests

url = 'http://localhost:5000/api'

r = requests.post(url,json={'total_credits':2, 'collections_12_mths_ex_med':4, 'terms':5, 'annual_inc':4, 'numb_credit':2, 'delinq_2yrs':8, 'Experience':5, 'pub_rec':5, 'acc_now_delinq':5,'debt_income_ratio':2, 'Rate_of_intrst':4, 'mths_since_last_major_derog':4, 'tot_colle_amt':9, 'tot_curr_bal':8, 'loan_amnt':7, 'total_rec_late_fee':3, 'mths_since_last_record':4, 'mths_since_last_delinq':1, 'total_rec_int':2})
print(r.json())











