Contract Churn Prediction: End to End Machine Learning Project
Project Overview

This project builds an end to end predictive churn model to identify contracts at risk of cancellation across a portfolio of 999 contracts and 296 customers. Using customer firmographic data and contract level information the model predicts the binary outcome of whether a contract will churn or be retained — enabling proactive retention intervention before revenue is lost.

The project was developed as part of a data science full analytical lifecycle from raw PDF data extraction through to a deployable machine learning model and stakeholder presentation.

Business Problem

A 29% contract churn rate was identified across the portfolio resulting in $38,431 in lost revenue — representing 28.3% of total portfolio value. The core challenge is that churn is invisible until it is too late — by the time a customer cancels the decision has already been made.

This project addresses the timing problem by building a model that scores every active contract with a churn probability — giving the retention team a prioritised list of at-risk contracts to act on before cancellation occurs.

Key Findings


46% of customers are partially churned — they have lost some but not all contracts and represent the highest revenue recovery opportunity at approximately $90,000
Low revenue companies churn at 41.2% — more than double the rate of High revenue companies at 17.5%
Gold tier contracts have the highest churn rate at 40% despite not being the most expensive tier
Revenue related features dominate the model — contract revenue, customer total revenue and the engineered interaction feature Revenue Per Company Size are the top three predictors



Datasets

Two datasets were provided in PDF format:

DatasetDescriptionSizeContract DataContract ID, Product tier, Revenue, Churn Flag999 contractsFirmographic DataCustomer profile — Tenure, Industry, Company Revenue, Analytics Intent Score296 customers

The datasets were joined on Customer ID using a LEFT JOIN to preserve all 999 contract rows.

contract-churn-prediction/
│
├── data/
│   ├── contracts_clean.csv          # Cleaned contract data
│   ├── firmographic_clean.csv       # Cleaned firmographic data
│   ├── merged_clean.csv             # Left joined merged dataset
│   ├── final_clean.csv              # Final clean dataset post treatment
│   ├── model_ready.csv              # Feature engineered dataset (21 features)
│   └── churn_scores.csv             # Model output with churn probability scores
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb       # Phase 1 — Data extraction and cleaning
│   ├── 02_eda.ipynb                 # Phase 2 — Exploratory data analysis
│   ├── 03_feature_engineering.ipynb # Phase 3 — Feature engineering
│   └── 04_model_building.ipynb      # Phase 4 — Model building and evaluation
│
├── presentation/
│   └── Churn_Prediction_Presentation.pptx  # Stakeholder presentation
│
├── outputs/
│   ├── model_comparison.csv         # Model performance metrics
│   ├── feature_importance.csv       # Feature importance scores
│   ├── confusion_matrix_data.csv    # Confusion matrix breakdown
│   └── roc_curve_data.csv           # ROC curve data for all models
│
└─


Methodology

Phase 1 — Data Cleaning


Extracted raw data from PDF files using layout-preserving text extraction and regex pattern matching
Performed full data quality audit — duplicate checks, data type validation, outlier detection and missing value treatment
Joined two datasets on Customer ID using a LEFT JOIN preserving all 999 contract rows


Phase 2 — Exploratory Data Analysis


Statistical analysis of all numerical and categorical variables
Churn rate analysis across product tier, industry, company revenue band, tenure and analytics intent score
Customer level analysis — identified 46% partially churned customers as the highest priority retention segment
Revenue at risk analysis — $38,431 lost to churn across the portfolio
Correlation analysis — no single numeric variable strongly predicts churn alone confirming the need for a machine learning model


Phase 3 — Feature Engineering

Expanded from 9 raw columns to 21 model ready features across 5 groups:

Feature GroupFeatures CreatedMethodOne Hot EncodingProduct_Gold, Product_Silver, Product_Platinum, Industry_Banking, Industry_Commercial, Industry_Other, Company Revenue_High, Company Revenue_Low, Company Revenue_Mediumpd.get_dummies()Customer LevelTotal_Contracts, Customer_Total_Revenue, Is_Multi_Contractgroupby + transformContract RevenueRevenue_Log, Is_Zero_Revenuenp.log1p() and boolean flagInteraction FeatureRevenue_Per_Company_SizeRevenue / Company size numeric scaleTenure BandsTenure_Early, Tenure_Mid, Tenure_Maturepd.cut() then one hot encoding

Most impactful engineered feature: Revenue_Per_Company_Size — captures how financially significant a contract is relative to the size of the company paying for it. Ranked 3rd in feature importance.

Phase 4 — Model Building

Three classification algorithms were trained and compared on an 80/20 stratified train/test split:

ModelAccuracyPrecisionRecallF1 ScoreAUC-ROCLogistic Regression59.5%37.6%60.3%46.4%0.666Random Forest64.5%47.3%44.8%46.1%0.662Gradient Boosting74.0%66.7%24.1%35.9%0.690

Champion Model: Logistic Regression

Selected based on:


Highest Recall at 60.3% — catches 3x more actual churners than Gradient Boosting
Best F1 Score at 46.4% — most balanced model overall
Highest interpretability — coefficients explain exactly which features drive each prediction
Marginal AUC difference of only 0.024 from Gradient Boosting does not justify 3x fewer churners caught



Churn Risk Scoring

The champion model outputs a churn probability score between 0 and 1 for every contract. Contracts are segmented into three actionable risk bands:

Risk BandThresholdContractsRecommended ActionHigh Risk> 60%38Immediate retention outreach within 48 hoursMedium Risk30% – 60%151Schedule check-in within 2 weeksLow Risk< 30%11Monitor quarterly — no immediate action


Tools and Technologies

ToolPurposePython 3.8+Data cleaning, EDA, feature engineering, model buildingpandasData manipulation and analysisnumpyNumerical operations and transformationsscikit-learnModel training, evaluation and preprocessingmatplotlib / seabornData visualisationSAS Visual AnalyticsInteractive EDA dashboards and chartsSAS Model StudioModel pipeline validation and comparisonJupyter NotebookDevelopment environment
