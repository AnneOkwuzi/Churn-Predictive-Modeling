#!/usr/bin/env python
# coding: utf-8

# # Investigation and Preliminary Analysis

# ###### First importing the neccessary libaries to commence EDA on the Firmographic and Contract datasets.  This stage is the general inspection, cleaning and merging of the two datasets 

# In[4]:


#Importing neccesarry libaries and loading datasets
import pandas as pd
import numpy as np

contracts    = pd.read_csv('Contract Data 1.csv')
firmographic = pd.read_csv('Firmographic Data 1.csv')


# ###### Initial inspection of the datasets

# In[5]:


df1 = contracts
df1.shape 


# In[6]:


df2 = firmographic
df2.shape 


# In[7]:


df1.dtypes


# In[8]:


df2.dtypes


# In[9]:


df1.head()


# In[10]:


df2.head()


# In[11]:


df1.describe()


# In[12]:


df2.describe()


# In[13]:


# Unique values for all columns in the contract dataset
for col in df1.columns:
    print(f"{col}: {df1[col].unique()}\n")


# In[14]:


# Unique values for all columns in the Frimographic dataset
for col in df2.columns:
    print(f"{col}: {df2[col].unique()}\n")


# In[15]:


#Checking for null values
df1.isnull()


# In[16]:


#Checking for null values
df2.isnull()


# In[17]:


#Checking for duplicate values
df1.duplicated()


# In[18]:


#Checking for duplicate values
df2.duplicated()


# Merging both datasets as a LEFT JOIN: This preserves all the 999 rows in the contract dataset regardless of whether a firmographic match exists.

# In[19]:


# Performing a left join
merge1 = df1.merge(df2, on='Customer ID', how='left')
merge1


# In[20]:


# This code downloads the new merged dataset to local drive
merge1.to_csv('custom_merge.csv', index=False)


# In[21]:


# Checking for NaN values in the newly merged dataset on the column that was joined
merge1[merge1['Customer ID'].isnull()]


# In[22]:


merge1.isnull().sum()


# In[23]:


merge1[merge1['Tenure'].isnull()]


# In[24]:


# Getting statistical information of the new merged dataset for different columns (features) 
merge1[['Revenue', 'Tenure', 'Analytics Intent Score']].describe().round(2)


# ##### For the firmographic revenue: 
# 
# The mean shows us that the average company contract revenue is worth $136, the std of 117.01 shows us  that most companies fall within $117 which is below the mean. The cheapest contract is worth $0. 
# 25% of all contracts are worth $46 or less. 
# 50% which is half of all contracts are worth $95 or less. This is the median. 
# 75% of contracts are worth $210 or less. Only the top 25% are above $210
# The most expensive contract is worth $493
# What this tells us is that most contracts are clustered between $46 and $210. 

# ##### For the Tenure:
# 
# The average customer has been with the company for about 5 and a half years
# Some are brand new, some have been there for 10 years, the minimum tenure is 1, meaning the newest customers have been around for 1 year.
# 25%, means that one quarter of the customers have been with the company for 3 years or less
# 50%, means that half of the customers have been around for 5 years or less
# 75%,  means that three quarters of customers have been around for 8 years or less
# The longest serving customers have been around for 10 years
# What this tells us is that customers in the company have a reasonable balanced mix of new and established relationships.

# ##### For the analytics intent score:
# 
# The average intent score is about 5.6 out of 10, sitting just above the midpoint
# An std of 2.98 shows that the scores are spread widely across the full range. Customers vary a lot in how engaged they are
# Some customers had an intent score as low as 1 out of 10
# 25% of customers had an intent score of 3 
# 50% of customers had an intent score of 5 or below
# 75% of the customer had an intent score of 8 or below
# Some customers had an intent score of exactly 10
# This tells us that the intent scores are spread fairly evenly from low to high with no strong skew in either direction. There is a good mix of low and high analytics intent.

# In[25]:


merge1.describe(include="all")


# In[26]:


# Statistics of the categorical features


categorical_cols = ['Product', 'Industry', 'Company Revenue']

for col in categorical_cols:
    print("=" * 50)
    print(f"COLUMN: {col}")
    print("=" * 50)
    print(f"  Total count        : {merge1[col].count()}")
    print(f"  Unique values      : {merge1[col].nunique()}")
    print(f"  Mode (most common) : {merge1[col].mode()[0]}")
    print(f"\n  Value counts:")
    vc  = merge1[col].value_counts()
    pct = merge1[col].value_counts(normalize=True).mul(100).round(2)
    summary = pd.DataFrame({'count': vc, '%': pct})
    print(summary.to_string())
    print()


# ##### For product tier; 
# Gold is the most popular product tier with 397 contracts (39.7%), followed closely by Silver at 364 contracts (36.4%), while Platinum is the least common at 238 contracts (23.8%). 
# 
# ##### Looking at industry;
# Other is the largest group at 401 contracts (40.1%), Commercial accounts for 356 contracts (35.6%), and Banking makes up the remaining 242 contracts (24.2%). 
# 
# ##### For company revenue;
# The portfolio skews toward smaller companies. Low revenue customers hold the largest number at 405 contracts (40.5%), Medium revenue customers account for 353 contracts (35.3%), and High revenue customers represent the smallest group at 241 contracts (24.1%). Overall the distribution across all three variables is reasonably balanced, with no single category overwhelmingly dominating the others.

# # For visuals (just in case)
# 
# import pandas as pd
# import numpy as np
# 
# df = pd.read_csv('final_clean.csv')
# 
# # ───────────────────────────────────────────────────────────────
# # DESCRIPTIVE STATISTICS — CONTINUOUS NUMERIC COLUMNS ONLY
# # ───────────────────────────────────────────────────────────────
# 
# continuous_cols = ['contract_revenue', 'tenure', 'intent_score']
# 
# for col in continuous_cols:
#     print("=" * 50)
#     print(f"COLUMN: {col}")
#     print("=" * 50)
#     print(f"  Count  : {df[col].count()}")
#     print(f"  Mean   : {df[col].mean():.2f}")
#     print(f"  Median : {df[col].median():.2f}")
#     print(f"  Mode   : {df[col].mode()[0]}")
#     print(f"  Std Dev: {df[col].std():.2f}")
#     print(f"  Min    : {df[col].min()}")
#     print(f"  Max    : {df[col].max()}")
#     print(f"  Skew   : {df[col].skew():.2f}")
#     print()

# In[28]:


# Calculating the correlation

numeric_cols = ['Revenue', 'Tenure', 'Analytics Intent Score', 'Churn Flag']
corr_matrix = merge1[numeric_cols].corr().round(2)

print(corr_matrix)


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import os

plt.figure(figsize=(14, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


# The results of the correlation heatmap reveals that none of the numeric variables alone show a strong correlation with churn.
# 
# Analytics Intent Score has the highest correlation with Churn Flag at 0.09, indicating a very weak positive relationship meaning customers with slightly higher intent scores are marginally more likely to churn, which is counterintuitive but consistent with the churn rate analysis findings. Revenue shows a correlation of -0.02 with Churn Flag, which is essentially zero, confirming that contract revenue value alone is not a reliable predictor of whether a contract will churn. Tenure similarly shows a near zero correlation of 0.01 with Churn Flag, suggesting that how long a customer has been with the company does not linearly predict churn on its own. The weak correlations observed across all variables do not mean these variables do not provide any valuable information, rather they suggest that churn is driven by combinations of factors rather than any single variable in isolation, which is precisely why a machine learning model is needed to capture these more complex relationships.

# ## Customer Level Analysis
# 
# Each customer has more than one contract, therefore, we need to need to understand churn rates on a customer level. Customer who churned ALL their contracts is a total relationship loss. A customer who churned SOME contract is a partial loss. A customer who churned NONE is fully retained. These three groups need different strategies.

# In[30]:


# Finding the number of contracts per customer

contracts_per_customer = merge1.groupby('Customer ID')['Contract ID'].count().reset_index()
contracts_per_customer.columns = ['Customer ID', 'Number of Contracts']
contracts_per_customer = contracts_per_customer.sort_values('Number of Contracts', ascending=False)
print(contracts_per_customer.to_string(index=False))


# In[31]:


# Distribution of contracts per customer
dist = contracts_per_customer['Number of Contracts'].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor('white')
    ax.set_title(title, color='black', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color='black', fontsize=9)
    ax.set_ylabel(ylabel, color='black', fontsize=9)
    ax.tick_params(colors='black')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('black')

# Chart 1 — Distribution
axes[0].bar(dist.index, dist.values, color='#4472C4',
            edgecolor='white', linewidth=0.8)
for x, y in zip(dist.index, dist.values):
    axes[0].text(x, y + 0.5, str(y), ha='center',
                 color='black', fontsize=8, fontweight='bold')
style_ax(axes[0], 'Distribution — Contracts per Customer',
         'Number of Contracts', 'Number of Customers')

# Chart 2 — Top 20 customers
top20 = contracts_per_customer.head(20)
axes[1].barh(top20['Customer ID'].astype(str),
             top20['Number of Contracts'],
             color='#ED7D31', edgecolor='white', linewidth=0.8)
for i, val in enumerate(top20['Number of Contracts']):
    axes[1].text(val + 0.1, i, str(val), va='center',
                 color='black', fontsize=8, fontweight='bold')
style_ax(axes[1], 'Top 20 Customers by Number of Contracts',
         'Number of Contracts', 'Customer ID')

plt.suptitle('Customer Contract Analysis',
             color='black', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# The output shows us in Chart 1 tells the story at a portfolio level, most customers are small. Chart 2 gives us insight on an account level, it shows that a few customers are disproportionately large and must be protected. Together they give us the big picture and the specific accounts to focus retention efforts on.
# 
# The contracts per customer analysis was conducted to understand the distribution of account sizes, identify the most valuable customer relationships, assess total relationship loss risk for single contract customers, and derive a new feature for the predictive model

# ### Creating New Data Assets

# In[32]:


# 

merge1.columns = ['Customer ID', 'Contract ID', 'Product', 'Revenue',
                  'Churn Flag', 'Tenure', 'Industry', 'Company Revenue',
                  'Analytics Intent Score']

customer_summary = merge1.groupby('Customer ID').agg(
    Total_Contracts     = ('Contract ID',  'count'),
    Total_Revenue       = ('Revenue',      'sum'),
    Avg_Revenue         = ('Revenue',      'mean'),
    Churned_Contracts   = ('Churn Flag',   'sum'),
    Retained_Contracts  = ('Churn Flag',   lambda x: (x==0).sum()),
    Customer_Churn_Rate = ('Churn Flag',   'mean'),
    Tenure              = ('Tenure',       'first'),
    Industry            = ('Industry',     'first'),
    Company_Revenue     = ('Company Revenue', 'first'),
    Intent_Score        = ('Analytics Intent Score', 'first')
).reset_index()

customer_summary['Customer_Churn_Rate'] = (
    customer_summary['Customer_Churn_Rate'] * 100
).round(2)

# Add churn category column
customer_summary['Churn_Category'] = customer_summary['Customer_Churn_Rate'].apply(
    lambda x: 'Fully Churned' if x == 100 
    else ('Partially Churned' if x > 0 else 'Fully Retained')
)

# Add single vs multi contract column
customer_summary['Contract_Type'] = customer_summary['Total_Contracts'].apply(
    lambda x: 'Single Contract' if x == 1 else 'Multi Contract'
)


print(f"Saved — {len(customer_summary)} customer rows")
print(customer_summary.head())


# In[33]:


customer_summary


# In[34]:


# This code downloads the new customer summary dataset to local drive
customer_summary.to_csv('customer_summary.csv', index=False)


# ## Feature Engineering

# The model will be learning from these new features
# 
# Original Numeric : Revenue, Tenure, Analytics Intent Score
# ProductProduct_Gold, Product_Platinum, Product_Silver
# 
# IndustryIndustry_Banking, Industry_Commercial, Industry_Other
# Company RevenueCompany Revenue_High, Company Revenue_Low, Company Revenue_Medium
# CustomerTotal_Contracts, Customer_Total_Revenue, Is_Multi_Contract
# Contract RevenueRevenue_Log, Is_Zero_Revenue
# InteractionRevenue_Per_Company_Size
# Tenure BandsTenure_Early, Tenure_Mid, Tenure_Mature

# In[35]:


# One Hot Encoding Categorical Variables
df = pd.get_dummies(merge1,
                    columns=['Product', 'Industry', 'Company Revenue'],
                    drop_first=False,
                    dtype=int)

print("categorical columns encoded")
print(f"Shape: {df.shape}")


# In[36]:


# Customer level feature engineering
df['Total_Contracts'] = df.groupby(
    'Customer ID')['Contract ID'].transform('count')

df['Customer_Total_Revenue'] = df.groupby(
    'Customer ID')['Revenue'].transform('sum')

df['Is_Multi_Contract'] = (df['Total_Contracts'] > 1).astype(int)

print("customer level features added")
print(f"  Total_Contracts range      : {df['Total_Contracts'].min()} to {df['Total_Contracts'].max()}")
print(f"  Customer_Total_Revenue     : ${df['Customer_Total_Revenue'].min()} to ${df['Customer_Total_Revenue'].max()}")
print(f"  Is_Multi_Contract counts   : {df['Is_Multi_Contract'].value_counts().to_dict()}")


# In[37]:


# Revenue per company size feature engineering 
# Map company revenue band to numeric scale
revenue_map = {'Low': 1, 'Medium': 2, 'High': 3}
df['Company_Revenue_Numeric'] = merge1['Company Revenue'].map(revenue_map)

# Contract revenue divided by company size scale
df['Revenue_Per_Company_Size'] = (
    df['Revenue'] / df['Company_Revenue_Numeric']
).round(2)

# Drop the intermediate numeric column — no longer needed
df.drop(columns=['Company_Revenue_Numeric'], inplace=True)

print("revenue interaction feature added")
print(f"  Revenue_Per_Company_Size range: {df['Revenue_Per_Company_Size'].min():.2f} to {df['Revenue_Per_Company_Size'].max():.2f}")


# In[38]:


# Tenure band feature engineering
df['Tenure_Band'] = pd.cut(
    df['Tenure'],
    bins=[0, 3, 6, 10],
    labels=['Early', 'Mid', 'Mature']
)

tenure_dummies = pd.get_dummies(df['Tenure_Band'],
                                prefix='Tenure',
                                drop_first=False,
                                dtype=int)

df = pd.concat([df, tenure_dummies], axis=1)
df.drop(columns=['Tenure_Band'], inplace=True)

print("tenure band features added")
print(f"  Tenure_Early  : {df['Tenure_Early'].sum()} contracts")
print(f"  Tenure_Mid    : {df['Tenure_Mid'].sum()} contracts")
print(f"  Tenure_Mature : {df['Tenure_Mature'].sum()} contracts")


# In[39]:


# Dropping ID columns from the 
df_model = df.drop(columns=['Customer ID', 'Contract ID'])


# In[40]:


# Handling null values
# Impute numeric nulls with median
null_cols = df_model.columns[df_model.isnull().any()].tolist()
print(f"Null columns found: {null_cols}")

for col in null_cols:
    median_val = df_model[col].median()
    df_model[col] = df_model[col].fillna(median_val)
    print(f"  {col} — filled with median: {median_val:.2f}")

print(f"\n  Nulls remaining: {df_model.isnull().sum().sum()}")


# In[41]:


#Final Overview
target   = 'Churn Flag'
features = [c for c in df_model.columns if c != target]

print("\n" + "=" * 55)
print("FINAL FEATURE SET")
print("=" * 55)
print(f"  Shape          : {df_model.shape}")
print(f"  Total features : {len(features)}")
print(f"  Target         : {target}")
print(f"  Nulls          : {df_model.isnull().sum().sum()}")
print(f"\n  All features:")
for f in features:
    print(f"    {f}")


# In[42]:


df_model


# In[43]:


# Saving new dataset for model development to local drive

df_model.to_csv('df_model.csv', index=False)


# ## Churn Predictive Model

# In[44]:


# Importing useful libraries for model development

import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[45]:


# ── STEP 1: LOAD MODEL READY DATA ─────────────────────────────

target   = 'Churn Flag'
features = [c for c in df_model.columns if c != target]

X = df_model[features]
y = df_model[target]

print(f"Shape        : {df_model.shape}")
print(f"Features     : {len(features)}")
print(f"Churn rate   : {y.mean()*100:.1f}%")
print(f"Class counts : {y.value_counts().to_dict()}")


# In[46]:


# ── STEP 2: TRAIN TEST SPLIT ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set : {X_train.shape[0]} rows")
print(f"Test set     : {X_test.shape[0]} rows")
print(f"Train churn  : {y_train.mean()*100:.1f}%")
print(f"Test churn   : {y_test.mean()*100:.1f}%")


# In[47]:


# ── STEP 3: SCALE FEATURES FOR LOGISTIC REGRESSION ────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── STEP 4: BUILD THREE MODELS ─────────────────────────────────
# Model 1 — Logistic Regression
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)
lr_model.fit(X_train_scaled, y_train)
print("Regression trained")

# Model 2 — Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)
print("Random Forest trained")

# Model 3 — Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)
print("Gradient Boosting trained")

# ── STEP 5: EVALUATE ALL THREE MODELS ─────────────────────────
models = {
    'Logistic Regression' : (lr_model, X_test_scaled),
    'Random Forest'       : (rf_model, X_test),
    'Gradient Boosting'   : (gb_model, X_test)
}

results = {}

for name, (model, X_eval) in models.items():
    y_pred      = model.predict(X_eval)
    y_pred_prob = model.predict_proba(X_eval)[:, 1]

    results[name] = {
        'Accuracy' : accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall'   : recall_score(y_test, y_pred),
        'F1 Score' : f1_score(y_test, y_pred),
        'AUC-ROC'  : roc_auc_score(y_test, y_pred_prob),
        'y_pred'   : y_pred,
        'y_prob'   : y_pred_prob
    }


# In[48]:


# ── STEP 6: MODEL COMPARISON TABLE ────────────────────────────
comparison = pd.DataFrame({
    name: {
        'Accuracy'  : f"{res['Accuracy']*100:.1f}%",
        'Precision' : f"{res['Precision']*100:.1f}%",
        'Recall'    : f"{res['Recall']*100:.1f}%",
        'F1 Score'  : f"{res['F1 Score']*100:.1f}%",
        'AUC-ROC'   : f"{res['AUC-ROC']:.4f}"
    }
    for name, res in results.items()
})

print("\nMODEL COMPARISON")
print(comparison.to_string())

best_model_name = max(results, key=lambda x: results[x]['AUC-ROC'])
print(f"Best model: {best_model_name}")
print(f"   AUC-ROC  : {results[best_model_name]['AUC-ROC']:.4f}")


# In[49]:


# ── STEP 7: CONFUSION MATRICES ─────────────────────────────────
for name, res in results.items():
    cm = confusion_matrix(y_test, res['y_pred'])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{name} — Confusion Matrix")
    print(f"  True Negatives  (correctly retained) : {tn}")
    print(f"  True Positives  (correctly churned)  : {tp}")
    print(f"  False Positives (wrong churn alarm)  : {fp}")
    print(f"  False Negatives (missed churners)    : {fn}")


# In[50]:


# ── STEP 8: CROSS VALIDATION ───────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (model, _) in models.items():
    X_cv   = X_train_scaled if name == 'Logistic Regression' else X_train
    scores = cross_val_score(model, X_cv, y_train,
                             cv=cv, scoring='roc_auc')
    print(f"\n{name} — Cross Validation")
    print(f"  CV AUC scores : {[round(s,4) for s in scores]}")
    print(f"  Mean AUC      : {scores.mean():.4f}")
    print(f"  Std Dev       : {scores.std():.4f}")


# In[51]:


# ── STEP 9: FEATURE IMPORTANCE ─────────────────────────────────
# Random Forest
rf_importance = pd.DataFrame({
    'Feature'   : features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRANDOM FOREST — Top 10 Features:")
print(rf_importance.head(10).to_string(index=False))


# In[52]:


# Gradient Boosting
gb_importance = pd.DataFrame({
    'Feature'   : features,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nGRADIENT BOOSTING — Top 10 Features:")
print(gb_importance.head(10).to_string(index=False))


# In[53]:


# Logistic Regression
lr_importance = pd.DataFrame({
    'Feature'    : features,
    'Coefficient': abs(lr_model.coef_[0])
}).sort_values('Coefficient', ascending=False)

print("\nLOGISTIC REGRESSION — Top 10 Features:")
print(lr_importance.head(10).to_string(index=False))


# In[54]:


# ── STEP 10: CHURN PROBABILITY SCORES ─────────────────────────
churn_scores = pd.DataFrame({
    'Actual_Churn'     : y_test.values,
    'Churn_Probability': lr_model.predict_proba(X_test_scaled)[:, 1].round(4),
    'Prediction'       : lr_model.predict(X_test_scaled)
}).reset_index(drop=True)

churn_scores['Risk_Band'] = pd.cut(
    churn_scores['Churn_Probability'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

print("\nRisk band distribution:")
print(churn_scores['Risk_Band'].value_counts().to_string())

print("\nTop 15 highest risk contracts:")
print(churn_scores.nlargest(15, 'Churn_Probability').to_string(index=False))


# In[55]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

fig = plt.figure(figsize=(18, 20))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('white')
    ax.set_title(title, color='black', fontsize=10,
                 fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color='black', fontsize=8)
    ax.set_ylabel(ylabel, color='black', fontsize=8)
    ax.tick_params(colors='black', labelsize=8)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# In[56]:


# ── ROW 1: CONFUSION MATRICES ──────────────────────────────────
for idx, (name, (model, X_eval)) in enumerate(models.items()):
    ax = fig.add_subplot(gs[0, idx])
    cm = confusion_matrix(y_test, model.predict(X_eval))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Retained', 'Churned'],
                yticklabels=['Retained', 'Churned'],
                ax=ax, cbar=False,
                annot_kws={'size': 12, 'weight': 'bold'})
    style_ax(ax, f'Confusion Matrix\n{name}',
             'Predicted', 'Actual')


# In[57]:


# ── ROW 2: ROC CURVES + FEATURE IMPORTANCE ─────────────────────
# ROC Curves
ax_roc = fig.add_subplot(gs[1, 0])
colors = ['#4472C4', '#ED7D31', '#70AD47']
for (name, (model, X_eval)), color in zip(models.items(), colors):
    y_prob       = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _  = roc_curve(y_test, y_prob)
    auc          = roc_auc_score(y_test, y_prob)
    ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC={auc:.3f})')
ax_roc.plot([0,1],[0,1],'k--', linewidth=1, label='Random (AUC=0.5)')
ax_roc.set_xlabel('False Positive Rate', color='black', fontsize=8)
ax_roc.set_ylabel('True Positive Rate', color='black', fontsize=8)
ax_roc.set_title('ROC Curves — All Models', color='black',
                  fontsize=10, fontweight='bold')
ax_roc.legend(fontsize=7, loc='lower right')
ax_roc.spines['top'].set_visible(False)
ax_roc.spines['right'].set_visible(False)


# In[58]:


# Random Forest Feature Importance
ax_rf = fig.add_subplot(gs[1, 1])
rf_top = rf_importance.sort_values('Importance').tail(10)
ax_rf.barh(rf_top['Feature'], rf_top['Importance'],
           color='#ED7D31', edgecolor='white')
style_ax(ax_rf, 'Random Forest\nFeature Importance (Top 10)',
         'Importance', '')


# In[59]:


# Gradient Boosting Feature Importance
ax_gb = fig.add_subplot(gs[1, 2])
gb_top = gb_importance.sort_values('Importance').tail(10)
ax_gb.barh(gb_top['Feature'], gb_top['Importance'],
           color='#70AD47', edgecolor='white')
style_ax(ax_gb, 'Gradient Boosting\nFeature Importance (Top 10)',
         'Importance', '')


# In[60]:


# ── ROW 3: MODEL COMPARISON BAR CHARTS ─────────────────────────


metric_names = ['Accuracy', 'Precision', 'Recall']
model_names  = ['Log. Reg.', 'Rand. Forest', 'Grad. Boost']
bar_colors   = ['#4472C4', '#ED7D31', '#70AD47']

for idx, metric in enumerate(metric_names):
    ax = fig.add_subplot(gs[2, idx])
    values = [results[m][metric] for m in results.keys()]
    bars   = ax.bar(model_names, values,
                    color=bar_colors, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val*100:.1f}%',
                ha='center', color='black',
                fontsize=8, fontweight='bold')
    ax.set_ylim(0, 1)
    style_ax(ax, metric, '', metric)
    ax.tick_params(axis='x', labelsize=7)

plt.suptitle('Phase 4 — Model Building Results',
             color='black', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

fig = plt.figure(figsize=(18, 20))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('white')
    ax.set_title(title, color='black', fontsize=10,
                 fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color='black', fontsize=8)
    ax.set_ylabel(ylabel, color='black', fontsize=8)
    ax.tick_params(colors='black', labelsize=8)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── ROW 1: CONFUSION MATRICES ──────────────────────────────────
for idx, (name, (model, X_eval)) in enumerate(models.items()):
    ax = fig.add_subplot(gs[0, idx])
    cm = confusion_matrix(y_test, model.predict(X_eval))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Retained', 'Churned'],
                yticklabels=['Retained', 'Churned'],
                ax=ax, cbar=False,
                annot_kws={'size': 12, 'weight': 'bold'})
    style_ax(ax, f'Confusion Matrix\n{name}',
             'Predicted', 'Actual')

# ── ROW 2: ROC CURVES + FEATURE IMPORTANCE ─────────────────────
ax_roc = fig.add_subplot(gs[1, 0])
colors = ['#4472C4', '#ED7D31', '#70AD47']
for (name, (model, X_eval)), color in zip(models.items(), colors):
    y_prob      = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc         = roc_auc_score(y_test, y_prob)
    ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC={auc:.3f})')
ax_roc.plot([0,1],[0,1],'k--', linewidth=1, label='Random (AUC=0.5)')
ax_roc.set_xlabel('False Positive Rate', color='black', fontsize=8)
ax_roc.set_ylabel('True Positive Rate', color='black', fontsize=8)
ax_roc.set_title('ROC Curves — All Models', color='black',
                  fontsize=10, fontweight='bold')
ax_roc.legend(fontsize=7, loc='lower right')
ax_roc.spines['top'].set_visible(False)
ax_roc.spines['right'].set_visible(False)

# Random Forest Feature Importance
ax_rf = fig.add_subplot(gs[1, 1])
rf_top = rf_importance.sort_values('Importance').tail(10)
ax_rf.barh(rf_top['Feature'], rf_top['Importance'],
           color='#ED7D31', edgecolor='white')
style_ax(ax_rf, 'Random Forest\nFeature Importance (Top 10)',
         'Importance', '')

# Gradient Boosting Feature Importance
ax_gb = fig.add_subplot(gs[1, 2])
gb_top = gb_importance.sort_values('Importance').tail(10)
ax_gb.barh(gb_top['Feature'], gb_top['Importance'],
           color='#70AD47', edgecolor='white')
style_ax(ax_gb, 'Gradient Boosting\nFeature Importance (Top 10)',
         'Importance', '')

# ── ROW 3: MODEL COMPARISON BAR CHARTS ─────────────────────────
metric_names = ['Accuracy', 'Precision', 'Recall']
model_names  = ['Log. Reg.', 'Rand. Forest', 'Grad. Boost']
bar_colors   = ['#4472C4', '#ED7D31', '#70AD47']

for idx, metric in enumerate(metric_names):
    ax     = fig.add_subplot(gs[2, idx])
    values = [results[m][metric] for m in results.keys()]
    bars   = ax.bar(model_names, values,
                    color=bar_colors, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val*100:.1f}%',
                ha='center', color='black',
                fontsize=8, fontweight='bold')
    ax.set_ylim(0, 1)
    style_ax(ax, metric, '', metric)
    ax.tick_params(axis='x', labelsize=7)

plt.suptitle('Model Building Results',
             color='black', fontsize=14, fontweight='bold')
plt.tight_layout()


# In[66]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, recall_score

# ── ROC CURVE — STANDALONE ─────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 8))
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

colors = ['#4472C4', '#ED7D31', '#70AD47']
for (name, (model, X_eval)), color in zip(models.items(), colors):
    y_prob      = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc         = roc_auc_score(y_test, y_prob)
    ax1.plot(fpr, tpr, color=color, linewidth=2,
             label=f'{name} (AUC={auc:.3f})')

ax1.plot([0,1],[0,1], 'k--', linewidth=1, label='Random (AUC=0.500)')
ax1.set_xlabel('False Positive Rate', color='black', fontsize=11)
ax1.set_ylabel('True Positive Rate', color='black', fontsize=11)
ax1.set_title('ROC Curve For All Models', color='black',
               fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='lower right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color('#cccccc')
ax1.spines['left'].set_color('#cccccc')
ax1.tick_params(colors='black')
plt.tight_layout()
plt.show()


# In[67]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score

model_names = list(models.keys())
bar_colors  = ['#4472C4', '#ED7D31', '#70AD47']

# ── AUC-ROC BAR CHART ─────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 8))
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

auc_values = [roc_auc_score(y_test, model.predict_proba(X_eval)[:, 1])
              for name, (model, X_eval) in models.items()]

bars1 = ax1.bar(model_names, auc_values,
                color=bar_colors, edgecolor='white', width=0.5)

for bar, val in zip(bars1, auc_values):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f'{val:.3f}',
             ha='center', color='black',
             fontsize=11, fontweight='bold')

ax1.set_xlabel('Model', color='black', fontsize=11)
ax1.set_ylabel('AUC-ROC Score', color='black', fontsize=11)
ax1.set_title('AUC-ROC Score by Model',
               color='black', fontsize=13, fontweight='bold')
ax1.set_ylim(0.5, 0.75)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color('#cccccc')
ax1.spines['left'].set_color('#cccccc')
ax1.tick_params(colors='black', labelsize=9)
plt.tight_layout()
plt.show()


# In[69]:


fig2, ax2 = plt.subplots(figsize=(10, 8))
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')

model_names  = list(models.keys())
recall_values = [recall_score(y_test, model.predict(X_eval)) * 100
                 for name, (model, X_eval) in models.items()]
bar_colors   = ['#4472C4', '#9DC3E6', '#ED7D31']

bars = ax2.bar(model_names, recall_values,
               color=bar_colors, edgecolor='white', width=0.5)

for bar, val in zip(bars, recall_values):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.8,
             f'{val:.1f}%',
             ha='center', color='black',
             fontsize=11, fontweight='bold')

ax2.set_xlabel('Model', color='black', fontsize=11)
ax2.set_ylabel('Recall (%)', color='black', fontsize=11)
ax2.set_title('Correctly Identified Recall Churners',
               color='black', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 80)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('#cccccc')
ax2.spines['left'].set_color('#cccccc')
ax2.tick_params(colors='black')
plt.tight_layout()
plt.show()


# In[62]:


# ── 1. MODEL COMPARISON TABLE ──────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

comparison_df = pd.DataFrame({
    'Model'    : ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'Accuracy' : [results['Logistic Regression']['Accuracy']*100,
                  results['Random Forest']['Accuracy']*100,
                  results['Gradient Boosting']['Accuracy']*100],
    'Precision': [results['Logistic Regression']['Precision']*100,
                  results['Random Forest']['Precision']*100,
                  results['Gradient Boosting']['Precision']*100],
    'Recall'   : [results['Logistic Regression']['Recall']*100,
                  results['Random Forest']['Recall']*100,
                  results['Gradient Boosting']['Recall']*100],
    'F1_Score' : [results['Logistic Regression']['F1 Score']*100,
                  results['Random Forest']['F1 Score']*100,
                  results['Gradient Boosting']['F1 Score']*100],
    'AUC_ROC'  : [results['Logistic Regression']['AUC-ROC'],
                  results['Random Forest']['AUC-ROC'],
                  results['Gradient Boosting']['AUC-ROC']]
})
comparison_df.to_csv('comparison_df.csv', index=False)


# ── 2. CONFUSION MATRIX DATA ───────────────────────────────────
cm_data = []
for name, res in results.items():
    cm       = confusion_matrix(y_test, res['y_pred'])
    tn, fp, fn, tp = cm.ravel()
    cm_data.append({'Model': name, 'Category': 'True Negative',  'Count': tn})
    cm_data.append({'Model': name, 'Category': 'True Positive',  'Count': tp})
    cm_data.append({'Model': name, 'Category': 'False Positive', 'Count': fp})
    cm_data.append({'Model': name, 'Category': 'False Negative', 'Count': fn})

cm_df = pd.DataFrame(cm_data)
cm_df.to_csv('cm_df.csv', index=False)


# ── 3. FEATURE IMPORTANCE ──────────────────────────────────────
rf_importance['Model'] = 'Random Forest'
gb_importance['Model'] = 'Gradient Boosting'
lr_importance.columns  = ['Feature', 'Importance']
lr_importance['Model'] = 'Logistic Regression'

feature_imp_df = pd.concat([
    rf_importance.head(10),
    gb_importance.head(10),
    lr_importance.head(10)
], ignore_index=True)

feature_imp_df.to_csv('feature_imp_df.csv', index=False)

# ── 4. CHURN SCORES ────────────────────────────────────────────
churn_scores.to_csv('churn_scores.csv', index=False)


# ── 5. ROC CURVE DATA ──────────────────────────────────────────
roc_data = []
for name, (model, X_eval) in models.items():
    y_prob      = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    for f, t in zip(fpr, tpr):
        roc_data.append({
            'Model': name,
            'FPR'  : round(f, 4),
            'TPR'  : round(t, 4)
        })

roc_df = pd.DataFrame(roc_data)
roc_df.to_csv('roc_curve_data.csv', index=False)


# In[70]:


# Logistic Regression feature importance via coefficients
lr_importance = pd.DataFrame({
    'Feature'    : features,
    'Importance' : abs(lr_model.coef_[0])
}).sort_values('Importance', ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

bars = ax.barh(lr_importance['Feature'],
               lr_importance['Importance'],
               color='#4472C4', edgecolor='white')

for bar, val in zip(bars, lr_importance['Importance']):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', color='black',
            fontsize=9, fontweight='bold')

ax.set_xlabel('Coefficient Magnitude', color='black', fontsize=11)
ax.set_title('Logistic Regression — Top 10 Feature Importance\n(Champion Model)',
             color='black', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#cccccc')
ax.spines['left'].set_color('#cccccc')
ax.tick_params(colors='black', labelsize=9)
plt.tight_layout()
plt.show()


# In[71]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Count contracts per risk band
risk_counts = churn_scores['Risk_Band'].value_counts().reindex(
    ['Low Risk', 'Medium Risk', 'High Risk']
)

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

bar_colors = ['#02C39A', '#ED7D31', '#E05252']
bars = ax.bar(risk_counts.index, risk_counts.values,
              color=bar_colors, edgecolor='white', width=0.5)

for bar, val in zip(bars, risk_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            str(val),
            ha='center', color='black',
            fontsize=12, fontweight='bold')

ax.set_xlabel('Risk Band', color='black', fontsize=11)
ax.set_ylabel('Number of Contracts', color='black', fontsize=11)
ax.set_title('Contracts by Risk Band — Logistic Regression\n(Test Set)',
             color='black', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#cccccc')
ax.spines['left'].set_color('#cccccc')
ax.tick_params(colors='black')
plt.tight_layout()
plt.show()


# In[72]:


churn_scores.to_csv('churn_scores.csv', index=False)


# In[ ]:




