# Customer Churn Prediction

A machine learning project for predicting **customer churn** using a
telecom customer dataset. The notebook performs data exploration,
missing-value handling, categorical encoding, model training,
classification evaluation, ROC-AUC comparison, and a sample prediction
for a new customer.

## Project Overview

Customer churn prediction helps identify customers who are likely to
leave a service. In this project, the target variable is `CHURN`, with
two classes:

-   `Yes` -- customer churn
-   `No` -- customer does not churn

The notebook compares multiple machine learning classification models
and evaluates them using accuracy, classification reports, confusion
matrices, and ROC-AUC.

## Dataset

The notebook uses:

``` text
Baza customer Telecom v2.csv
```

The dataset contains **8,453 records** and **14 columns**.

### Main Features

  Feature                    Description
  -------------------------- ----------------------------------
  `PID`                      Customer identifier
  `CRM_PID_Value_Segment`    Customer value segment
  `EffectiveSegment`         Customer/business segment
  `Billing_ZIP`              Billing ZIP/postal code
  `KA_name`                  KA/category identifier
  `Active_subscribers`       Number of active subscribers
  `Not_Active_subscribers`   Number of inactive subscribers
  `Suspended_subscribers`    Number of suspended subscribers
  `Total_SUBs`               Total subscribers
  `AvgMobileRevenue`         Average mobile revenue
  `AvgFIXRevenue`            Average fixed-line revenue
  `TotalRevenue`             Total revenue
  `ARPU`                     Average Revenue Per User
  `CHURN`                    Target variable indicating churn

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   LightGBM
-   XGBoost
-   Google Colab

## Machine Learning Models

The notebook trains and evaluates the following models:

1.  Support Vector Machine (SVM)
2.  Logistic Regression
3.  Random Forest
4.  LightGBM
5.  Gradient Boosting
6.  XGBoost

## Workflow

``` text
Telecom Customer Dataset
          ↓
      Data Loading
          ↓
   Exploratory Analysis
          ↓
    Missing Value Check
          ↓
 Missing Value Imputation
          ↓
   Duplicate Check
          ↓
 Categorical Encoding
          ↓
 Train/Test Split
          ↓
   Model Training
          ↓
 Model Evaluation
          ↓
 ROC-AUC Comparison
          ↓
 New Customer Prediction
```

## Data Exploration

The notebook first loads the dataset and examines:

-   First rows of the dataset
-   Data types
-   Descriptive statistics
-   Missing values
-   Duplicate records

The dataset contains missing values in several columns, including:

-   `CRM_PID_Value_Segment`
-   `Billing_ZIP`
-   `Not_Active_subscribers`
-   `Suspended_subscribers`
-   `ARPU`

The notebook fills these missing values using the **mode** of the
corresponding column.

## Data Preprocessing

### Missing Values

Missing values are filled using the most frequent value:

``` python
df['CRM_PID_Value_Segment'].fillna(
    df['CRM_PID_Value_Segment'].mode()[0],
    inplace=True
)
```

The same approach is applied to the other columns containing missing
values.

### Categorical Encoding

Categorical/object columns are converted into numerical values using
`LabelEncoder`.

``` python
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
```

### Train/Test Split

The data is split into training and testing sets using:

``` python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This produces an 80/20 train-test split.

## Model Evaluation

The models are evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion Matrix
-   ROC-AUC

### Accuracy Results

  Model                   Accuracy
  --------------------- ----------
  SVM                       94.80%
  Logistic Regression       94.80%
  Random Forest             94.74%
  LightGBM                  94.86%
  Gradient Boosting         94.68%
  XGBoost                   94.38%

### ROC-AUC Results

  Model                   ROC-AUC
  --------------------- ---------
  SVM                      0.4921
  Logistic Regression      0.5596
  Random Forest            0.6037
  LightGBM                 0.5516
  Gradient Boosting        0.5837
  XGBoost                  0.5299

The notebook uses the ROC-AUC results to select **Random Forest** for
the example new-customer prediction because it had the highest ROC-AUC
among the models evaluated in the notebook.

> **Important:** The reported accuracy is high while the churn-class
> recall is very low. The test set contains 1,603 non-churn cases and
> only 88 churn cases, and several models predict almost all
> observations as non-churn. Therefore, accuracy alone should not be
> used to judge churn-detection performance.

## Example Confusion Matrix Results

For Random Forest:

``` text
[[1602    1]
 [  88    0]]
```

This shows that the model correctly classified most non-churn cases but
did not correctly identify the churn cases in the recorded test output.

LightGBM produced:

``` text
[[1602    1]
 [  86    2]]
```

XGBoost produced:

``` text
[[1594    9]
 [  86    2]]
```

These results highlight the class-imbalance challenge present in the
dataset.

## ROC-AUC Visualization

The notebook calculates ROC curves and AUC values for all six models and
creates a horizontal bar chart comparing their ROC-AUC scores.

The comparison includes:

``` text
SVM
Logistic Regression
Random Forest
LightGBM
Gradient Boosting
XGBoost
```

## New Customer Prediction

The notebook also demonstrates how to make a prediction for a custom
customer record.

Example input includes:

-   Customer ID
-   Customer value segment
-   Effective segment
-   Billing ZIP
-   KA name
-   Active subscribers
-   Inactive subscribers
-   Suspended subscribers
-   Total subscribers
-   Mobile revenue
-   Fixed revenue
-   Total revenue
-   ARPU

The recorded example produced:

``` text
Predicted churn probability: 0.27
Predicted class: No Churn
```

The example uses the Random Forest model for the final prediction.

## Project Structure

``` text
Customer-Churn-analysis/
│
├── Churn_prediction.ipynb
├── README.md
└── data/
    └── dataset_link.txt
```

The original dataset is not included in the repository in the notebook
itself; the notebook loads the CSV from Google Drive.

## How to Run

### Option 1 -- Google Colab

Open the notebook in Google Colab and run the cells sequentially.

The notebook already contains an **Open in Colab** link.

Before running the notebook, make sure the dataset is available at:

``` text
/content/drive/MyDrive/Baza customer Telecom v2.csv
```

### Option 2 -- Local Jupyter Notebook

Install the required libraries:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn lightgbm xgboost
```

Then open:

``` text
Churn_prediction.ipynb
```

Update the dataset path in the notebook to the location of your local
CSV file.

## Key Takeaways

-   The project performs end-to-end customer churn classification.
-   Six machine learning models are trained and evaluated.
-   Missing values are handled using mode imputation.
-   Categorical features are encoded using `LabelEncoder`.
-   Model performance is compared using ROC-AUC in addition to accuracy.
-   The notebook demonstrates prediction on a custom customer record.
-   The recorded results show that the dataset has a substantial
    class-imbalance issue for churn prediction.

## Possible Improvements

The current notebook can be extended by:

-   Applying SMOTE to address class imbalance.
-   Using stratified train/test splitting.
-   Applying feature scaling consistently during model training.
-   Using separate `LabelEncoder` objects for each categorical feature.
-   Using `OneHotEncoder` or a preprocessing pipeline for categorical
    variables.
-   Tuning model hyperparameters.
-   Using cross-validation for more robust evaluation.
-   Optimizing for churn recall, F1-score, PR-AUC, or ROC-AUC instead of
    accuracy alone.
-   Performing feature-importance analysis.
-   Adding explainability using SHAP.
-   Building a deployment interface for individual customer churn
    prediction.

## Conclusion

This project demonstrates a complete machine learning workflow for
telecom customer churn prediction, from data loading and preprocessing
to model comparison and individual customer prediction.

The results also demonstrate why evaluating a churn model requires more
than accuracy: identifying the minority churn class is an important part
of the problem.

## Author

**Deepthi Pachigulla**

B.Tech -- Computer Science and Engineering (Data Science)

**Skills demonstrated:** Python \| Pandas \| NumPy \| Scikit-learn \|
Machine Learning \| Classification \| Data Analysis \| Matplotlib \|
Seaborn \| LightGBM \| XGBoost

## License

This project is intended for educational and portfolio purposes. Dataset
ownership and licensing remain with the original dataset provider.
