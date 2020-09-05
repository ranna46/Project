# Multi-linear regression Project
Prediction of revolving balance of customer for deriving individual marketing strategies
Business Objective:
  Revolving credit means you're borrowing against a line of credit. Let's say a lender extends a certain amount of credit to you, against which you can borrow  repeatedly. The amount of credit you're allowed to use each month is your credit line, or credit limit. You're free to use as much or as little of that credit line as you wish on any purchase you could make with cash. Its just like a credit card and only difference is they have lower interest rate and they are secured by business assets.
  At the end of each statement period, you receive a bill for the balance. If you don't pay it off in full, you carry the balance, or revolve it, over to the next month and pay interest on any remaining balance. As you pay down the balance, more of your credit line becomes available and usually its useful for small loans
  As a bank or investor who are into this revolving balance here they can charge higher interest rates and convenience fees as there is lot of risk associated in customer paying the amount. Our company wants to predict the revolving balance maintained by the customer so that they can derive marketing strategies individually.

Data Set Details:

<This dataset consists of 2300 observations>

0)	-member_id unique ID assigned to each member
1)	loan_amnt loan amount ($) applied by the member
2)	terms:  term of loan (in months)
3)	-batch_ID batch numbers allotted to members
4)	Rate_of_intrst:  interest rate (%) on loan
5)	-Grade:grade assigned by the bank
6)	-sub_grade: grade assigned by the bank
7)	-emp_designation job / Employer title of member
8)	Experience: employment length, where 0 means less than one year and 10 means ten or more years
9)	-home_ownership status of home ownership
10)	annual_inc: annual income ($) reported by the member
11)	-verification_status status of income verified by the bank
12)	-purpose purpose of loan
13)	-State: living state of member
14)	debt-to-income ratio : ratio of member's total monthly debt
15)	Delinquency of past 2 years:  ( failure to pay an outstanding debt by due date)
16)	inq_6mths: Inquiries made in past 6 months
17)	total_months_delinq : number of months since last delinq
18)	Nmbr_months_last_record: number of months since last public record
19)	Numb_credit_lines:number of open credit line in member's credit line
20)	pub_rec number of derogatory public records
21)	Tota_credit_revolving_balance: total credit revolving balance
22)	total_credits: total number of credit lines available in members credit line
23)	list_status unique listing status of the loan - W(Waiting),F(Forwarded)
24)	int_rec: Total interest received till date
25)	late_fee_rev: Late fee received till date
26)	recov_chrg: post charge off gross recovery
27)	collection_recovery_fee post charge off collection fee
28)	exc_med_colle_12mon: number of collections in last 12 months excluding medical collections
29)	since_last_major_derog: months since most recent 90 day or worse rating
30)	-application_type indicates when the member is an individual or joint
31)	-verification_status_joint indicates if the joint members income was verified by the bank
32)	last_pay_week: indicates how long (in weeks) a member has paid EMI after batch enrolled
33)	nmbr_acc_delinq: number of accounts on which the member is delinquent
34)	colle_amt: total collection amount ever owed
35)	curr_bal: total current balance of all accounts
