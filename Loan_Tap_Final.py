# ### Context:
# 
# - A Non-Banking Finance Company like LoanTap is an online platform committed to delivering customized loan products to millennials. 
# - They innovate in an otherwise dull loan segment, to deliver instant, flexible loans on consumer friendly terms to salaried professionals and businessmen.
# 
# 
# - The data science team is building an underwriting layer to determine the creditworthiness of MSMEs as well as individuals.
# 
# - Company deploys formal credit to salaried individuals and businesses 4 main financial instruments:
# 
#     - Personal Loan
#     - EMI Free Loan
#     - Personal Overdraft
#     - Advance Salary Loan
# 
# - This case study will focus on the underwriting process behind Personal Loan only
# 
# 
# 
# ## Problem Statement:
# 
# - Given a set of attributes for an Individual, determine if a credit line should be extended to them. If so, what should the repayment terms be in business recommendations?
# 
# ####  Tradeoff Questions:
# 
# - How can we make sure that our model can detect real defaulters and there are less false positives? This is important as we can lose out on an opportunity to finance more individuals and earn interest on it.
# 
# - Since NPA (non-performing asset) is a real problem in this industry, it’s important we play safe and shouldn’t disburse loans to anyone
# 
# 
# ## Data dictionary:
# 
# 1. loan_amnt : The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
# 2. term : The number of payments on the loan. Values are in months and can be either 36 or 60.
# 3. int_rate : Interest Rate on the loan
# 4. installment : The monthly payment owed by the borrower if the loan originates.
# 5. grade : Institution assigned loan grade
# 6. sub_grade : Institution assigned loan subgrade
# 7. emp_title :The job title supplied by the Borrower when applying for the loan.*
# 8. emp_length : Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
# 9. home_ownership : The home ownership status provided by the borrower during registration or obtained from the credit report.
# 10. annual_inc : The self-reported annual income provided by the borrower during registration.
# 11. verification_status : Indicates if income was verified by Institution, not verified, or if the income source was verified
# 12. issue_d : The month which the loan was funded
# 13. loan_status : Current status of the loan - Target Variable
# 14. purpose : A category provided by the borrower for the loan request.
# 15. title : The loan title provided by the borrower
# 16. dti : A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested Institution loan, divided by the borrower’s self-reported monthly income.
# 17. earliest_cr_line :The month the borrower's earliest reported credit line was opened
# 18. open_acc : The number of open credit lines in the borrower's credit file.
# 19. pub_rec : Number of derogatory public records
# 20. revol_bal : Total credit revolving balance
# 21. revol_util : Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
# 22. total_acc : The total number of credit lines currently in the borrower's credit file
# 23. initial_list_status : The initial listing status of the loan. Possible values are – W, F
# 24. application_type : Indicates whether the loan is an individual application or a joint application with two co-borrowers
# 25. mort_acc : Number of mortgage accounts.
# 26. pub_rec_bankruptcies : Number of public record bankruptcies
# 27. Address: Address of the individual
# 
# 
# 
# 
# 
# 
# 
# 


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import figure

import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import t

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv("logistic_regression.txt")



df

df.shape

# - #### 396030 data points , 26 features , 1 label.

# ## Missing Values Check: 

def missing_df(data):
    total_missing_df = data.isna().sum().sort_values(ascending = False)
    percentage_missing_df = ((data.isna().sum()/len(data)*100)).sort_values(ascending = False)
    missingDF = pd.concat([total_missing_df, percentage_missing_df],axis = 1, keys=['Total', 'Percent'])
    return missingDF


missing_data = missing_df(df)
missing_data[missing_data["Total"]>0]


(df.isna().sum() / df.shape[0] ) * 100



# ### Descriptive Statistics : 



df.describe().round(1)

# - #### Loan Amount, Installments, Annual Income , revol_bal : all these columns have large differnece in mean and median . That means outliers are present in the data. 

df.nunique()

df.info()

columns_type = df.dtypes

columns_type[columns_type=="object"]

df.describe(include="object")

len(columns_type[columns_type=="object"])

26-15 


# - #### 15 Non-numerical (categorical/date time) features present in the dataset. 

df["loan_status"].value_counts(normalize=True)*100

# - #### As we can see, there is an imbalance in the data. 
# - 80% belongs to the class 0 : which is loan fully paid. 
# - 20% belongs to the class 1 : which were charged off. 



plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(method='spearman'), annot=True)
plt.show()



# ## loan_amnt :
# 
# - #### The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
# 

df.groupby(by = "loan_status")["loan_amnt"].describe()

plt.figure(figsize=(5,7))
sns.boxplot(y=df["loan_amnt"],
            x=df["loan_status"])

sns.histplot(df["loan_amnt"],bins = 15)

# - ####  for loan status Charged_off, the mean and median of loan_amount is higher than fully paid.
# - #### also the distribution of loan_amnt is right skewed, which says it has outlier presence. 
# 





# ##  term :
# 
# - #### The number of payments on the loan. Values are in months and can be either 36 or 60.
# 

df["term"].value_counts(dropna=False)

# ####  P[loan_statis | term]

pd.crosstab(index=df["term"],
            columns=df["loan_status"], normalize="index" , margins  = True
           ) * 100

pd.crosstab(index=df["term"],
            columns =df["loan_status"], normalize="columns"
           ).plot(kind = "bar")

# as we can observe 
# the conditional probability 
# of loan fully paid given that its 36 month term is higher then charged off.  

# loan fully paid probability when 60 month term is lower than charged off. 

term_values = {' 36 months': 36, ' 60 months': 60}
df['term'] = df['term'].map(term_values)












# ##  int_rate :
# 
# - #### Interest Rate on the loan
# 

df.groupby(by = "loan_status")["int_rate"].describe()

sns.histplot(df["int_rate"],bins = 15)

sns.boxplot(x=df["int_rate"],
            y=df["loan_status"])

df[df["loan_status"] == "Charged Off"]["int_rate"].median(),df[df["loan_status"] == "Charged Off"]["int_rate"].mean()


df[df["loan_status"] == "Fully Paid"]["int_rate"].median(),df[df["loan_status"] == "Fully Paid"]["int_rate"].mean()

# for charge_off Loan Status ,
# interest_rate median and mean is higher than fully paid.


# - ####  for loan status Charged_off, the mean and median of interest_rate is higher than fully paid.
# - #### also the distribution of interest_rate is right skewed, which says it has outlier presence. 
# 







# ##  grade :
# 
# - #### LoanTap assigned loan grade
# 
# - #### Loan grades are set based on both the borrower's credit profile and the nature of the contract.
# 

df["grade"].value_counts().sort_values().plot(kind = "bar")

df["grade"].value_counts(dropna=False)

pd.crosstab(index = df["grade"],
            columns= df["loan_status"],normalize= "index", margins = True)

pd.crosstab(index = df["grade"],
            columns= df["loan_status"],normalize= "columns").plot(kind  = "bar")

#  probability of loan_status as fully_paid decreases with grade is E,F,G



## we can conclude the relationship exists 
## between loan_status and LoanTap assigned loan grade.









# ## sub_grade : 
# 
# - #### LoanTap assigned loan subgrade
# 

# pd.crosstab(index = df["sub_grade"],
#             columns= df["loan_status"],normalize= "index", margins = True)*100

pd.crosstab(index = df["sub_grade"],
            columns= df["loan_status"],normalize= "columns", ).plot(kind = "bar")

# Similar pattern is observed for sub_grade as grade . 

#  later target encoding 











# ##   emp_title :
# 
# - #### The job title supplied by the Borrower when applying for the loan.*
# 

df["emp_title"].value_counts(dropna=False).sort_values(ascending=False).head(15)

df["emp_title"].nunique()

# missing values need to be treated with model based imputation .


# total unique job_titles are 173,105. 
# target encoding while creating model. 





# ##  emp_length :
# 
# - #### Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
# 

df["emp_length"].value_counts(dropna=False)

pd.crosstab(index = df["emp_length"],
            columns= df["loan_status"],normalize= "index", margins = True)*100

pd.crosstab(index = df["emp_length"],
            columns= df["loan_status"],normalize= "index").plot(kind = "bar")

# visually there doent seems to be much correlation between employement length 
# and loan_status.




stats.chi2_contingency(pd.crosstab(index = df["emp_length"],
            columns= df["loan_status"]))



# ##  home_ownership : 
# 
# - #### The home ownership status provided by the borrower during registration or obtained from the credit report.
# 

df["home_ownership"].value_counts(dropna=False)

df["home_ownership"] = df["home_ownership"].replace({"NONE":"OTHER", "ANY":"OTHER"})

pd.crosstab(index = df["home_ownership"],
            columns= df["loan_status"],normalize= "index", margins = True)*100

pd.crosstab(index = df["home_ownership"],
            columns= df["loan_status"],normalize= "index").plot(kind= "bar")

# visually there doent seems to be much correlation between home_ownership 
# and loan_status.
# later target encoding or label encoding .










# ## annual_inc :
# 
# - #### The self-reported annual income provided by the borrower during registration.
# 

sns.distplot(df["annual_inc"])

df["annual_inc"].describe()

sns.distplot(np.log(df[df["annual_inc"]>0]["annual_inc"]))

plt.figure(figsize=(5,7))
sns.boxplot(y=np.log(df[df["annual_inc"]>0]["annual_inc"]),
            x=df["loan_status"])

##from above boxplot, there seems to be no difference between annual income,
# for loan status categories 














# ##  verification_status : 
# 
# - #### Indicates if income was verified by LoanTap, not verified, or if the income source was verified
# 

df["verification_status"].value_counts(dropna=False)

pd.crosstab(index = df["verification_status"],
            columns= df["loan_status"],normalize= "index", margins = True)*100

pd.crosstab(index = df["verification_status"],
            columns= df["loan_status"],normalize= "index").plot(kind = "bar")





# later  label encoding  
# .
# Verified           1
# Source Verified    2
# Not Verified       0














# ## purpose :
# - #### A category provided by the borrower for the loan request.
# 

df["purpose"].nunique()



print(df["purpose"].value_counts(dropna=False))
pd.crosstab(index = df["purpose"],
            columns= df["loan_status"],normalize= "index", margins = True)*100
pd.crosstab(index = df["purpose"],
            columns= df["loan_status"],normalize= "index").plot(kind = "bar")


(df["purpose"].value_counts(dropna=False,normalize=True)).plot(kind = "bar")


# ### 13. 
# 
# ###  loan_status : Current status of the loan - Target Variable
# 

df["loan_status"].value_counts(dropna=False).plot(kind = "bar")


df["loan_status"].value_counts(dropna=False, normalize=True)  * 100

# Imbalanced data. 

# 80% loans are fully paid.
# 20% loans are charged_off 



#     ## most of the loans are taken for 
#         debit_card,
#         dept_consolidation , 
#         home_improvement and others category. 
#     ## number of loan applications and amount per purpose category are highest in  above category.
# 
#  











# ##  title :
# 
# - #### The loan title provided by the borrower
# 
# 
# 

df["title"].nunique()

df["title"]

# title and purpose are in a way same features. 
# later needs to drop this feature. 




# ##  dti :
# - #### A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LoanTap loan, divided by the borrower’s self-reported monthly income.
# 

#     dti = monthly total dept payment / monthly income excluding mortgages

df["dti"].describe()

sns.boxenplot((df["dti"]))

# looks like there are lots of outliers in dti column .

plt.figure(figsize=(5,7))
sns.boxplot(y=np.log(df[df["dti"]>0]["dti"]),
            x=df["loan_status"])



#      issue_d :
#     The month which the loan was funded¶



# ### issue_d :
# - #### The month which the loan was funded
# 

# df["issue_d"].value_counts(dropna=False)

# later use in feature engineering ! 

# ## earliest_cr_line :
# - #### The month the borrower's earliest reported credit line was opened
# 

df["Loan_Tenure"] = ((pd.to_datetime(df["issue_d"]) -pd.to_datetime(df["earliest_cr_line"]))/np.timedelta64(1, 'M'))

# pd.to_datetime(df["earliest_cr_line"])

# The month which the loan was funded

# pd.to_datetime(df["issue_d"])

 sns.histplot(((pd.to_datetime(df["issue_d"]) -pd.to_datetime(df["earliest_cr_line"]))/np.timedelta64(1, 'M')))




plt.figure(figsize=(5,7))
sns.boxplot(y=np.log(((pd.to_datetime(df["issue_d"]) -pd.to_datetime(df["earliest_cr_line"]))/np.timedelta64(1, 'M'))),
            x=df["loan_status"])





# ## open_acc : 
# 
# - #### The number of open credit lines in the borrower's credit file.
# 

df.groupby("loan_status")["open_acc"].describe()

df["open_acc"].nunique()

sns.histplot(df["open_acc"],bins = 25)


plt.figure(figsize=(5,7))
sns.boxplot(y= df["open_acc"],
            x=df["loan_status"])















# ## pub_rec : 
# 
# - #### Number of derogatory public records
# 
# 
# 
# - “Derogatory” is seen as negative to lenders, and can include late payments, collection accounts, bankruptcy, charge-offs and other negative marks on your credit report. This can impact your ability to qualify for new credit.

df.groupby("loan_status")["pub_rec"].describe()

plt.figure(figsize=(5,7))
sns.boxplot(y= df["pub_rec"],
            x=df["loan_status"])

print(df["pub_rec"].value_counts(dropna=False))
pd.crosstab(index = df["pub_rec"],
            columns= df["loan_status"],normalize= "index", margins = True)*100
pd.crosstab(index = df["pub_rec"],
            columns= df["loan_status"],normalize= "index").plot(kind = "bar")






# ## revol_bal : 
# 
# - #### Total credit revolving balance
# 
# 
# With revolving credit, a consumer has a line of credit he can keep using and repaying over and over. The balance that carries over from one month to the next is the revolving balance on that loan.
# 
# 
# 

df.groupby("loan_status")["revol_bal"].describe()



sns.histplot(np.log(df["revol_bal"]))


plt.figure(figsize=(5,7))
sns.boxplot(y= np.log(df["revol_bal"]),
            x=df["loan_status"])







# ##  revol_util :
# - #### Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
# 
# 
# Your credit utilization rate, sometimes called your credit utilization ratio, is the amount of revolving credit you're currently using divided by the total amount of revolving credit you have available. In other words, it's how much you currently owe divided by your credit limit. It is generally expressed as a percent.

df.groupby("loan_status")["revol_util"].describe()

plt.figure(figsize=(5,7))
sns.boxplot(y= np.log(df["revol_util"]),
            x=df["loan_status"])











# ##  total_acc : 
# 
# - #### The total number of credit lines currently in the borrower's credit file
# 

# df["total_acc"].value_counts()

df.groupby("loan_status")["total_acc"].describe()

plt.figure(figsize=(5,7))
sns.boxplot(y= np.log(df["total_acc"]),
            x=df["loan_status"])





# ## initial_list_status :
# 
# - #### The initial listing status of the loan. Possible values are – W, F
# 

df["initial_list_status"].value_counts()

print(df["initial_list_status"].value_counts(dropna=False))

pd.crosstab(index = df["initial_list_status"],
            columns= df["loan_status"],normalize= "columns").plot(kind = "bar")








# ## application_type : 
# 
# - #### Indicates whether the loan is an individual application or a joint application with two co-borrowers
# 

df["application_type"].value_counts()

print(df["application_type"].value_counts(dropna=False))

pd.crosstab(index = df["application_type"],
            columns= df["loan_status"],normalize= "index").plot(kind = "bar")


# ## mort_acc : 
# 
# - #### Number of mortgage accounts.
# 

# df["mort_acc"].value_counts(dropna=False)

df.groupby("loan_status")["mort_acc"].describe()

plt.figure(figsize=(5,7))
sns.boxplot(y= np.log(df["mort_acc"]),
            x=df["loan_status"])

pd.crosstab(index = df["mort_acc"],
            columns= df["loan_status"],normalize= "index").plot(kind = "bar")



# ## pub_rec_bankruptcies :
# - #### Number of public record bankruptcies
# 

df["pub_rec_bankruptcies"].value_counts()

print(df["pub_rec_bankruptcies"].value_counts(dropna=False))
print(pd.crosstab(index = df["pub_rec_bankruptcies"],
            columns= df["loan_status"],normalize= "index", margins = True)*100)
pd.crosstab(index = df["pub_rec_bankruptcies"],
            columns= df["loan_status"],normalize= "index").plot(kind = "bar")








# ## Address:
# 
# - #### Address of the individual

df["address"][10]

df["address"] = df["address"].str.split().apply(lambda x:x[-1])

df["address"].value_counts()

pd.crosstab(index = df["address"],
            columns= df["loan_status"],normalize= "index").plot(kind = "bar")


df["pin_code"] = df["address"]
df.drop(["address"],axis = 1  ,inplace=True)



# # dropping unimportant columns 



df.drop(["title","issue_d","earliest_cr_line","initial_list_status"],axis = 1, inplace=True)



df.drop(["pin_code"],axis=1,inplace=True)

df.drop(["Loan_Tenure"],axis=1,inplace=True)



# ## Missing value treatment

missing_data[missing_data["Percent"]>0]

from sklearn.impute import SimpleImputer
Imputer = SimpleImputer(strategy="most_frequent")
df["mort_acc"] = Imputer.fit_transform(df["mort_acc"].values.reshape(-1,1))



df.dropna(inplace=True)

missing_df(df)

# ## Pre-proccessing : 
# 

# ### Feature Engineering

from category_encoders import TargetEncoder

TE = TargetEncoder()



df["loan_status"].replace({"Fully Paid":0,
                          "Charged Off" : 1},inplace=True)

df.sample(3)

df.columns

target_enc = ["sub_grade","grade",'term', 'emp_title', 'emp_length', 'home_ownership', 'verification_status', 'purpose', 'application_type']

for col in target_enc:
    from category_encoders import TargetEncoder
    TEncoder = TargetEncoder()
    
    df[col] = TEncoder.fit_transform(df[col],df["loan_status"])

df

# ## Outlier treatment :



def outlier_remover(a,df):

    q1 = a.quantile(.25)
    q3 = a.quantile(.75)
    iqr = q3 - q1

    maxx = q3 + 1.5 * iqr
    minn = q1 - 1.5 * iqr

    return df.loc[(a>=minn) & (a<=maxx)]

floats = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'open_acc','revol_bal', 'revol_util', 'total_acc']
     

df.sample(3)

for i in floats:
    df = outlier_remover(df[i],df)

for i in floats:
    plt.figure(figsize=(15, 3))
    plt.subplot(121)
    sns.boxplot(y=df[i])
    plt.title(f"Boxplot of {i} before removing outliers")
    plt.subplot(122)
    sns.boxplot(y=df[i])
    plt.title(f"Boxplot of {i} after removing outliers")

    plt.show()

# # Missing value check : 

def missing_df(data):
    total_missing_df = data.isna().sum().sort_values(ascending = False)
    percentage_missing_df = ((data.isna().sum()/len(data)*100)).sort_values(ascending = False)
    missingDF = pd.concat([total_missing_df, percentage_missing_df],axis = 1, keys=['Total', 'Percent'])
    return missingDF


missing_data = missing_df(df)
missing_data[missing_data["Total"]>0]


df.columns

df.drop(["mort_acc","pub_rec_bankruptcies"],axis = 1 , inplace=True)

df.drop(["pub_rec"],axis = 1 , inplace=True)

plt.figure(figsize=(24,15))
sns.heatmap(df.corr(),annot=True,cmap='BrBG_r')

plt.show()

# ## Train-test split : 

X = df.drop(["loan_status"],axis = 1)
y = df["loan_status"]



from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,
                                                      random_state=3,
                                                      test_size=0.2)

# ### Logistic Regression on Non-Standardised Data : 

from sklearn.linear_model import LogisticRegression
LR1st = LogisticRegression(class_weight="Auto")

LR1st.fit(X_train,y_train)

LR1st.score(X_test,y_test)

from sklearn.metrics import f1_score,recall_score,precision_score

f1_score(y_test,LR1st.predict(X_test))

recall_score(y_test,LR1st.predict(X_test))

precision_score(y_test,LR1st.predict(X_test))



# ## Standardizing  - preprocessing

from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()



StandardScaler.fit(X_train)



X_train = StandardScaler.transform(X_train)
X_test = StandardScaler.transform(X_test)


from sklearn.linear_model import LogisticRegression
LR_Std = LogisticRegression(C=1.0)
LR_Std.fit(X_train,y_train)
print("Accuracy: ",LR_Std.score(X_test,y_test))
print("f1_score: ",f1_score(y_test,LR_Std.predict(X_test)))
print("recall_score: ",recall_score(y_test,LR_Std.predict(X_test)))
print("precision_score: ",precision_score(y_test,LR_Std.predict(X_test)))

pd.DataFrame(data=LR_Std.coef_,columns=X.columns).T

pd.DataFrame(data=LR_Std.coef_,columns=X.columns).T.plot(kind = "bar")

# ## Data Balancing : 

from imblearn.over_sampling import SMOTE

SmoteBL = SMOTE(k_neighbors=7) 

X_smote , y_smote = SmoteBL.fit_resample(X_train,y_train)

X_smote.shape,  y_smote.shape

# y_smote.value_counts()








from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(max_iter=1000,class_weight="balanced")

from sklearn.model_selection import cross_val_score

cross_val_score(estimator = LogReg,
                cv=5,
                X = X_smote,
                y = y_smote,
                scoring= "f1"
            
       )

cross_val_score(estimator = LogReg,
                cv=5,
                X = X_smote,
                y = y_smote,
                scoring= "precision"
            
       )

cross_val_score(estimator = LogReg,
                cv=5,
                X = X_smote,
                y = y_smote,
                scoring= "accuracy"
            
       )



cross_val_score(estimator = LogReg,
                cv=5,
                X = X_train,
                y = y_train,
                scoring= "precision"
            
       )



from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression(max_iter=1000,class_weight="balanced")

LogReg.fit(X= X_train ,y = y_train)

LogReg.score(X_test,y_test)



LogReg.coef_.round(2)

from sklearn.metrics  import confusion_matrix, f1_score, precision_score,recall_score
print(confusion_matrix(y_test, LogReg.predict(X_test)))
print(precision_score(y_test ,LogReg.predict(X_test)))
print(recall_score(y_test ,LogReg.predict(X_test)))
print(f1_score(y_test ,LogReg.predict(X_test)))









LogReg.coef_

df.drop(["loan_status"], axis = 1).columns

feature_importance = pd.DataFrame(index = df.drop(["loan_status"],
                                                  axis = 1).columns,
                                  data = LogReg.coef_.ravel()).reset_index()
feature_importance

plt.figure(figsize=(10,15))
sns.barplot(y = feature_importance["index"],
           x =  feature_importance[0])

LogReg.score(X_train,y_train)

LogReg.score(X_test,y_test)

plt.figure(figsize=(15,15))

sns.heatmap(df.corr().round(2),annot=True,square=True)

# ## Metrics : 



from sklearn.metrics  import confusion_matrix, f1_score, precision_score,recall_score
confusion_matrix(y_test, LogReg.predict(X_test))



precision_score(y_test ,LogReg.predict(X_test))

recall_score(y_test ,LogReg.predict(X_test))



pd.crosstab(y_test ,LogReg.predict(X_test))



recall_score(y_train ,LogReg.predict(X_train))

recall_score(y_test ,LogReg.predict(X_test))

f1_score(y_test ,LogReg.predict(X_test))

f1_score(y_train ,LogReg.predict(X_train))

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import fbeta_score

cm_display  = ConfusionMatrixDisplay(confusion_matrix= confusion_matrix(y_test, 
                                                          LogReg.predict(X_test)),display_labels=[False,True])
cm_display.plot()
plt.show()

# fbeta_score

cm_display  = ConfusionMatrixDisplay(confusion_matrix= confusion_matrix(y_train, 
                                                          LogReg.predict(X_train)),display_labels=[False,True])
cm_display.plot()
plt.show()









from sklearn.tree import DecisionTreeClassifier

DecisionTreeClassifier = DecisionTreeClassifier(max_depth=5, splitter="best",
                                               criterion="entropy",class_weight ="balanced")

DecisionTreeClassifier.fit(X_train,y_train)

DecisionTreeClassifier.score(X_test,y_test)

# DecisionTreeClassifier.score(X_smote,y_smote)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=30,max_depth=10,class_weight="balanced")

RF.fit(X_train,y_train)

RF.score(X_test,y_test)

feature_importance = pd.DataFrame(index = df.drop(["loan_status"],
                                                  axis = 1).columns,
                                  data = RF.feature_importances_.ravel()).reset_index()
feature_importance

plt.figure(figsize=(10,15))
sns.barplot(y = feature_importance["index"],
           x =  feature_importance[0])



from sklearn.metrics import precision_recall_curve

def precision_recall_curve_plot(y_test, pred_proba_c1):
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

    threshold_boundary = thresholds.shape[0]
    # plot precision
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--')
    # plot recall
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recalls')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    plt.xlabel('Threshold Value'); plt.ylabel('Precision and Recall Value')
    plt.legend(); plt.grid()
    plt.show()

precision_recall_curve_plot(y_test, LogReg.predict_proba(X_test)[:,1])


def precision_recall_curve_plot(y_test, pred_proba_c1):
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

    threshold_boundary = thresholds.shape[0]
    # plot precision
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    # plot recall
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recalls')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    plt.xlabel('Threshold Value'); plt.ylabel('Precision and Recall Value')
    plt.legend(); plt.grid()
    plt.show()

precision_recall_curve_plot(y_test, LogReg.predict_proba(X_test)[:,1])

from sklearn.metrics import roc_auc_score,roc_curve

logit_roc_auc = roc_auc_score(y_test, LogReg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, LogReg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

LogReg.predict_proba(X_test)

precision_recall_curve_plot(y_test, RF.predict_proba(X_test)[:,1])


precision_recall_curve_plot(y_test, DecisionTreeClassifier.predict_proba(X_test)[:,1])












from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)

def custom_predict(X, threshold):
    probs = model.predict_proba(X) 
    return (probs[:, 1] > threshold).astype(int)

new_preds = custom_predict(X=X_test, threshold=0.75)

model.score(X_test,y_test)

precision_score(y_test,new_preds)







# ## Inferences and Report : 
# 
# - 396030 data points , 26 features , 1 label.
# 
# 
# - 80% belongs to the class 0 : which is loan fully paid.
# - 20% belongs to the class 1 : which were charged off.
# 
# 
# - Loan Amount distribution / media is slightly higher for Charged_off loanStatus.
# 
# 
# - Probability of CHarged_off status is higher in case of 60 month term.
# 
# 
# - Interest Rate mean and media is higher for Charged_off LoanStatus. 
# 
# 
# 
# 
# - Probability of Charged_off LoanStatus is higher for Loan Grades are E ,F, G. 
# - G grade has the highest probability of having defaulter.
# - Similar pattern is visible in sub_grades probability plot.
# 
# 
# 
# - Employement Length has overall same probability of Loan_status as fully paid and defaulter.
# - That means Defaulters has no relation with their Emoployement length.
# 
# 
# 
# - For those borrowers who have rental home, has higher probability of defaulters.
# - borrowers having their home mortgage and owns have lower probability of defaulter. 
# 
# 
# 
# - Annual income median is lightly higher for those who's loan status is as fully paid. 
# 
# 
# - Somehow , verified income borrowers probability of defaulter is higher than those who are not verified by loan tap. 
# 
# 
# 
# - Most of the borrowers take loans for dept-consolidation and credit card payoffs. 
# - the probability of defaulters is higher in the small_business owner borrowers. 
# 
# 
# 
# - debt-to-income ratio is higher for defaulters.
# 
# 
# - number of open credit lines in the borrowers credit file is same as for loan status as fully paid and defaulters.
# 
# 
# 
# - Number of derogatory public records increases , the probability of borrowers declared as defaulters also increases 
# - aspecially for those who have higher than 12 public_records.
# 
# 
# 
# 
# 
# - Total credit revolving balance is almost same for both borrowers who had fully paid loan and declared defaulter
# - but Revolving line utilization rate is higher for defaulter borrowers.
# 
# 
# 
# 
# 
# - Application type Direct-Pay has higher probability of defaulter borrowers than individual and joint. 
# 
# 
# 
# - Number of public record bankruptcies increasaes ,   higher the probability of defaulters.
# 
# 
# 
# 
# 
# 
# - Most important features/ data for prediction , as per Logistic Regression, Decision tree classifier and Random Forest  model are : Employee Title, Loan Grade and Sub-Grade, Interest rate and dept-to-income ratio. 
# 
# 
# 
# 
# 
# 

# ### Actionable Insights & Recommendations
# -  We should try to keep the precision higher as possible compare to recall , and keep the false positive low.
# - that will help not to missout the opportopportunity to finance more individuals and earn interest on it. This we can achieve by setting up the higher threshold.  
# - Giving loans to those even having slightly higher probability of defaulter, we can maximise the earning , by this risk taking method. 
# 
# - and Since NPA is a real problem in the industry  , Company should more investigate and check for the proof of assets. Since it was observed in probability plot,  verified borrowers had higher probability of defaulters than non-varified. 
# - Giving loans to those who have no mortgage house of any owned property have higher probability of defaulter , giving loan to this category borrowers can be a problem of NPA. 
# 





