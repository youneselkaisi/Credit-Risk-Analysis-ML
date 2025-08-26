# Credit Risk Prediction Using Machine Learning


## Overview
This project builds and evaluates neural-network models to predict credit risk (default vs non-default) using borrower demographics, credit attributes, and loan features from a public Kaggle dataset. The workflow covers data exploration, cleaning, preprocessing, modelling with Multi-Layer Perceptrons (MLPs), and model evaluation.

Who would use this:  
1) Risk and underwriting teams to screen applications, price loans, and set approval thresholds  
2) Portfolio managers to estimate expected loss, segment risk tiers, and run scenario analysis  
3) Collections and servicing to prioritize accounts most likely to roll into delinquency  
4) Fintech and neobank product teams to build automated decisioning pipelines and real-time risk scoring

Business and financial use cases: 
1) Reduce default rates by flagging high-risk applicants pre-approval  
2) Optimize interest rate offers by aligning price to predicted risk  
3) Lower manual review costs through automated triage and clear confidence thresholds  
4) Improve portfolio performance forecasting by turning borrower features into PD (probability of default) estimates

## Dataset
Source  
[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data?select=credit_risk_dataset.csv)
  
Author: Lao Tse  
License: CC0 Public Domain

Shape after load: 
Rows 32,581  
Columns 12

Field meanings and abbreviations  
person_age = Age in years  
person_income = Annual income  
person_home_ownership = Home ownership category  
person_emp_length = Years employed  
loan_intent = Loan purpose category  
loan_grade = Risk grade A–G where A is lowest risk  
loan_amnt = Loan amount requested  
loan_int_rate = Interest rate  
loan_status = Target variable where 0 = non-default and 1 = default  
loan_percent_income = Loan amount divided by income  
cb_person_default_on_file = Has borrower previously defaulted Y or N  
cb_person_cred_hist_length = Credit history length in years

## Data Exploration
The notebook inspects datatypes, missing values, and descriptive statistics. Key initial observations  
1) person_emp_length has 895 missing values  
2) loan_int_rate has 3,116 missing values  
3) Target loan_status baseline rate approximately 21.8 percent defaults in the raw data  
4) Wide income range with long right tail up to 6,000,000 and extreme ages up to 144

## EDA visuals
![Breakdown of Homeownership](https://github.com/youneselkaisi/Credit-Risk-Analysis-ML/blob/main/Visuals/Home%20ownership%20breakdown.png)  
![Distribution of Loan Types](https://github.com/youneselkaisi/Credit-Risk-Analysis-ML/blob/main/Visuals/Distribution%20of%20loan%20types.png)  
![Distribution of Loan Grades](https://github.com/youneselkaisi/Credit-Risk-Analysis-ML/blob/main/Visuals/Distribution%20of%20loan%20grades.png)  
![Boxplot of Person Age](https://github.com/youneselkaisi/Credit-Risk-Analysis-ML/blob/main/Visuals/Boxplot%20ages.png)  
![Boxplot of Person Income](https://github.com/youneselkaisi/Credit-Risk-Analysis-ML/blob/main/Visuals/Boxplot%20income.png)  
![Loan Status Breakdown](https://github.com/youneselkaisi/Credit-Risk-Analysis-ML/blob/main/Visuals/loan%20status.png)

## Data Cleaning
Steps applied: 
1) Filled loan_int_rate missing values with the median interest rate from the column  
2) Dropped rows with missing person_emp_length to avoid imputing employment tenure with weak assumptions  
3) Removed age outliers where person_age > 80 
4) Removed income outliers where person_income > 100000 
5) Dropped the first row which contained an employment-length outlier after prior filtering

After cleaning the working dataset shape is 27,514 rows by 12 columns with zero missing values.


## Feature Encoding and Preparation
Label encoding was applied to categorical variables using scikit-learn’s LabelEncoder  
1) person_home_ownership encoded to integers where categories map in alphabetical order Mortgage 0 Other 1 Own 2 Rent 3  
2) loan_intent encoded to integers DebtConsolidation 0 Education 1 Homeimprovement 2 Medical 3 Personal 4 Venture 5  
3) loan_grade encoded to integers A 0 B 1 C 2 D 3 E 4 F 5 G 6  
4) cb_person_default_on_file encoded to integers N 0 Y 1

<img width="1644" height="513" alt="image" src="https://github.com/user-attachments/assets/8deb7c04-e41a-4632-9fd8-8caeb0dec80a" />


Rationale:
1) MLPs require numeric inputs  
2) Label encoding is acceptable for tree-free neural models when categories are ordinal or when one-hot expansion is not required for capacity or runtime reasons  
3) loan_grade is ordinal by design so integer encoding preserves its order semantics


All models are Multi-Layer Perceptrons (scikit-learn MLPClassifier)

## Test 1: Unscaled features  
- hidden_layer_sizes: 32,16  
- Activation: relu  
- Learning rate adaptive with learning_rate_init: 0.001  
- Max epochs: 300 with early_stopping True  

<img width="779" height="116" alt="image" src="https://github.com/user-attachments/assets/ad0fb133-a412-47b9-90d6-63e83779d148" />

<img width="657" height="418" alt="image" src="https://github.com/user-attachments/assets/dbe6709d-9ad8-4bf2-8129-2e39f6228dab" />


Interpretation  
1) Model learns meaningful signal without scaling but underfits relative to standardized tests  
2) Early stopping triggered after validation plateau indicating stable convergence

## Test 2: standardized features  
- Scaler StandardScaler fit on training set only then applied to test set  
- hidden_layer_sizes 32,16  
- Activation: relu  
- Max epochs: 200  

<img width="855" height="85" alt="image" src="https://github.com/user-attachments/assets/55ce0677-3f25-4fee-a6d3-5abec0f86eeb" />

<img width="794" height="591" alt="image" src="https://github.com/user-attachments/assets/dba60b61-e5f1-43b1-b32e-71e6cfdec23e" />

<img width="505" height="327" alt="image" src="https://github.com/user-attachments/assets/80abfe91-1b5a-4477-87ee-cc49e87521bc" />


Interpretation  
1) Standardization markedly improves optimization stability and accuracy  
2) Convergence warning at max_iter 200 indicates additional epochs or regularization tuning could squeeze marginal gains

## Test 3: standardized features and deeper network  
- hidden_layer_sizes 32,16,8  
- Activation: relu  
- Learning rate adaptive with learning_rate_init: 0.01  
- Max epochs: 200  

<img width="663" height="93" alt="image" src="https://github.com/user-attachments/assets/6d8d930f-8691-4734-ad4b-68819a766757" />


<img width="697" height="528" alt="image" src="https://github.com/user-attachments/assets/ac5211a0-32ec-42b0-9268-21d9d531eba1" />

<img width="625" height="340" alt="image" src="https://github.com/user-attachments/assets/60aed196-a03c-47a5-8827-618347bb94a5" />



Interpretation:  
1) Best overall generalization among the three setups  
2) High recall for non-default and lower recall for default indicating thresholding and class-balance strategies could improve sensitivity to defaults  
3) Learning-rate schedule and additional hidden layer help capture nonlinear interactions among income, grade, and percent-income


<b>Summary and Findings</b>

In this project, I built 3 different MLP classifiers to evaluate performance on a credit risk dataset. In the first test, I used a basic 2 layer network and applied adaptive learning, which dynamically adjusts the learning rate during training. The loss curve fluctuated and early stopping kicked in but, I reached a solid test accuracy of 82.7%, with only slight overfitting. For the second test, I implemented standard scaling, knowing that MLPs are sensitive to feature scale, giving precedence to features with greater magnitudes. I also removed adaptive learning to isolate its impact and saw a clear improvement in the model’s learning and performance, as accuracy increased to 91.2%, showing the importance of scaling. In my third and final test, I added an extra hidden layer, reintroduced adaptive learning, and increased the learning rate. The loss decreased slightly and accuracy improved again to 92.1%. Although I couldn’t completely eliminate the slight overfitting (which remained around 0.5–1%), I’m pleased with my results achieving 92% test accuracy shows a great classification model.

## Limitations and Next Steps
Limitations  
1) Label encoding for non-ordinal categoricals can impose artificial order; consider one-hot encoding in future iterations  
2) Class imbalance leads to lower recall on defaults; calibration and cost-sensitive learning are needed for production

Next steps  
1) Introduce class weighting or focal loss via alternative frameworks to improve default recall  
2) Tune decision threshold using precision-recall trade-offs aligned to business costs  
3) Add cross-validation and calibration curves to stabilize and interpret predicted probabilities  
4) Explore SHAP for local and global explanations to support credit policy reviews


## Libraries Used
- **Python 3.10+**
- **Pandas** (data cleaning & wrangling)  
- **NumPy** (numerical operations)  
- **Matplotlib & Seaborn** (visualizations, EDA)  
- **Scikit-learn** (MLP, model evaluation) 


## How to Reproduce
Clone the repository and install dependencies

```bash
git clone https://github.com/youneselkaisi/Credit-Risk-Analysis-ML.git
cd Credit-Risk-Prediction-ML
pip install -r requirements.txt
