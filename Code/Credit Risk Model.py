#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import matplotlib.pyplot as plt


# # Part 1: Data Exploration

# In[56]:


# importing dataset as dataframe

df = pd.read_csv("credit_risk_dataset.csv")
df


# <b>About the Dataset:</b>
# 
# Name: Credit Risk Dataset
# 
# Source: https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data?select=credit_risk_dataset.csv
# 
# Author: Lao Tse
# 
# License: CC0 Public Domain 
# 
# Fields:
# - person_emp_length	= Years employed
# - loan_grade = Perceived risk rating assigned to a loan from A-G (low risk-high risk)
# - loan_int_rate = Interest Rate
# - loan_percent_income = Loan/Income
# - cb_person_default_on_file = Has borrower defaulted before? (Y/N)
# - cb_person_cred_hist_length = Length of credit history in years
# - loan_status = 0 is non default 1 is default

# #### Data Exploration and Visualizations

# In[59]:


# shape of dataset
df.shape


# In[60]:


# viewing data types & non-nulls
df.info()


# In[61]:


# statistical overview of data
df.describe()


# In[66]:


# column names
df.columns


# In[68]:


# count of nulls
df.isnull().sum()


# In[70]:


# filling interest rate nulls with median rate 
df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)

# dropping employment length nulls
df = df.dropna(subset=["person_emp_length"])


# In[72]:


# updated null count
df.isnull().sum()


# In[74]:


# pie chart for home ownership 

# Grouping data and counting occurrences
data_counts = df["person_home_ownership"].value_counts()  # Count occurrences of each category

# Creating pie chart
plt.pie(data_counts, labels=data_counts.index, autopct='%1.1f%%', startangle=45)
plt.title("Breakdown of Homeownership")

plt.show()


# In[75]:


# Bar chart showing distribution of loan types 

# Counting occurrences of each category
class_counts = df["loan_intent"].value_counts()

plt.bar(class_counts.index, class_counts.values, color='navy',zorder=3)

# Labels and title

plt.title("Distribution of Loan Types")

# rotating x-axis labels
plt.xticks(rotation=40)

# adding grid
plt.grid(axis='y')

plt.show()


# In[76]:


# Bar chart showing distribution of loan Grades 

# Counting occurrences of each category
class_counts = df["loan_grade"].value_counts().sort_values(ascending=True)

plt.barh(class_counts.index, class_counts.values, color='darkgreen',zorder=3)

# Labels and title

plt.title("Distribution of Loan Grades")

# adding grid
plt.grid(axis='x')

plt.show()


# In[78]:


# boxplot for borrower ages 

# plotting box plot
plt.figure(figsize=(6, 4))
plt.boxplot(df['person_age'],vert=True)  

# adding titles and lables
plt.title('Boxplot of Person Age')
plt.ylabel('Age')

plt.show()


# In[80]:


# boxplot for borrower incomes

plt.boxplot(df['person_income'],vert=True)  

# lables and title
plt.title('Boxplot of Person Income')
plt.ylabel('Income')

# adding grid
plt.grid(axis='y')

plt.show()


# In[82]:


# pie chart showing loan status breakdown

# mapping values for labels
status_counts = df['loan_status'].map({0: 'Non-Default', 1: 'Default'}).value_counts()

# plotting pie chart
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%')
plt.title('Loan Status Breakdown')

plt.show()


# # Part 2: Data Preprocessing

# #### Preperation for learning and explanation

# In[87]:


# dropping age outliers over 80

df = df[df['person_age'] <= 80]


# In[89]:


# dropping income outliers for incomes above $100,000

df = df[df['person_income'] <= 100000]


# In[92]:


# dropping first row that contained emp_length outlier 
df = df.drop(df.index[0])
df.head()


# In[93]:


# shape after removing outliers 

df.shape


# In[96]:


# Encoding categorical data into numerical data 

from sklearn.preprocessing import LabelEncoder

# Initializing LabelEncoder
label_encoder = LabelEncoder()

# Label Encoding the home owenership column
df["person_home_ownership"] = label_encoder.fit_transform(df["person_home_ownership"]) # mortgage = 0, other = 1, own = 2, rent = 3 (based on alphabetical order)

# Label Encoding the loan intent column
df["loan_intent"] = label_encoder.fit_transform(df["loan_intent"]) # debt = 0, education = 1, home = 2, medical = 3, personal = 4, venture = 5

# Label Encoding the loan grade column
df["loan_grade"] = label_encoder.fit_transform(df["loan_grade"]) # A = 0, B = 1, C = 2, D = 3, E = 4, F = 5, G = 6

# Label Encoding the loan grade column
df["cb_person_default_on_file"] = label_encoder.fit_transform(df["cb_person_default_on_file"]) # N = 0, Y = 1


# In[97]:


df.head()


# #### Preprocessing Explanation:
# -  Dropped rows where person_age > 80 and person_income > 100,000
# These are extreme outliers that could skew model performance and don't provide useful predictive power
# 
# - Label encoded categorical features to make them usable by the ML model:
# 
#     - person_home_ownership encoded as Mortgage = 0, Other = 1, Own = 2, Rent = 3
# 
#     - loan_intent encoded as DebtConsolidation = 0, Education = 1, ..., Venture = 5
# 
#     - loan_grade encoded from A = 0 to G = 6 to maintain credit grade order
# 
#     - cb_person_default_on_file encoded as No = 0, Yes = 1 to indicate previous default history
# 
# - Encoding is important as machine learning models require numerical values and can't interpret text 

# ## Part 3: Model Selection and Evaluation

# In[100]:


# Importing libraries for machine learning 

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# In[101]:


# setting feature and lable sets

y = df["loan_status"]
X = df.drop(["loan_status"],axis=1)
print(X,y)


# #### Test 1:

# In[103]:


# splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=9)

# verifying split ratios
print(len(X_train))  
print(len(X_test))


# #### MLP Justification
# 
# - Performs well on medium to large structured datasets like this one (27k rows), especially with encoded categorical and numeric features
# 
# - Capable of capturing complex nonlinear relationships between variables such as income, loan grade, and loan status
# 
# - Learns patterns through multiple layers making it more flexible than linear models 
# 
# - Suitable for binary classification tasks like predicting loan default and works best when features are standardized, as done during preprocessing

# In[105]:


# creating MLP classifier 1

ann_clf1 = MLPClassifier(hidden_layer_sizes=(32,16),activation='relu',learning_rate_init=0.001,learning_rate='adaptive',max_iter=300,early_stopping=True,verbose=True)


# In[106]:


# fitting classifier

ann_clf1.fit(X_train,y_train)


# In[107]:


# plotting MLP loss curve

plt.plot(ann_clf1.loss_curve_)

# titles and lables
plt.title("MLP Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.show()


# In[108]:


# training set accuracy
ann_clf1.score(X_train,y_train)


# In[109]:


# test set accuracy
y_pred=ann_clf1.predict(X_test)
accuracy_score(y_test,y_pred)


# #### Test 2:

# In[113]:


# setting feature and lable sets

y = df["loan_status"]
X = df.drop(["loan_status"],axis=1)


# In[117]:


from sklearn.preprocessing import StandardScaler

# scaling features to avoid bias towards larger features
scaler = StandardScaler()

# splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# fitting standard scalar onto training set
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[119]:


# creating MLP classifier 2                                   
ann_clf2 = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=200, random_state=42, verbose=True)

# training model
ann_clf2.fit(X_train, y_train)


# In[120]:


# plotting MLP loss curve

plt.plot(ann_clf2.loss_curve_)

# titles and lables
plt.title("MLP Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.show()


# In[121]:


# training accuracy
ann_clf2.score(X_train,y_train)


# In[122]:


# test accuracy
y_pred=ann_clf2.predict(X_test)
accuracy_score(y_test,y_pred)


# #### Test 3:

# In[124]:


# setting feature and lable sets

y = df["loan_status"]
X = df.drop(["loan_status"],axis=1)


# In[125]:


scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[126]:


# Creating MLP classifier 3
ann_clf3 = MLPClassifier(hidden_layer_sizes=(32,16,8), activation="relu", max_iter=200, random_state=17,learning_rate_init=0.01,verbose=True,learning_rate="adaptive")
ann_clf3.fit(X_train, y_train)


# In[128]:


# plotting MLP loss curve

plt.plot(ann_clf3.loss_curve_)

# titles and lables
plt.title("MLP Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.show()


# In[132]:


# training accuracy
ann_clf3.score(X_train,y_train)


# In[135]:


# test accuracy
y_pred=ann_clf3.predict(X_test)
accuracy_score(y_test,y_pred)


# In[136]:


# printing classification report for model
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred)) 


# ## Part 4: Conclusion

# <b>Summary and Findings</b>
# 
# In this assignment, I built 3 different MLP classifiers to evaluate performance on a credit risk dataset. In the first test, I used a basic 2 layer network and applied adaptive learning, which dynamically adjusts the learning rate during training. The loss curve fluctuated and early stopping kicked in but, I reached a solid test accuracy of 82.7%, with only slight overfitting. For the second test, I implemented standard scaling, knowing that MLPs are sensitive to feature scale, giving precdence to features with greater magnitudes. I also removed adaptive learning to isolate its impact and saw a clear improvement in the model’s learning and performance, as accuracy increased to 91.2%, showing the importance of scaling. In my third and final test, I added an extra hidden layer, reintroduced adaptive learning, and increased the learning rate. The loss decreased slightly and accuracy improved again to 92.1%. Although I couldn’t completely eliminate the slight overfitting (which remained around 0.5–1%), I’m pleased with my results achieving 92% test accuracy shows a great classification model.
