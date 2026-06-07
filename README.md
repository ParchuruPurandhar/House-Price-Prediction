# House Price Prediction using Machine Learning

## Project Overview

House price prediction is one of the most common real-world regression problems in data science. Accurate property valuation helps buyers, sellers, real estate agencies, and financial institutions make informed decisions.

This project develops a machine learning model to predict house prices based on various property characteristics such as location, area, number of bedrooms, bathrooms, and other housing features.

The objective is to build a robust predictive model that can estimate property prices with high accuracy and support data-driven real estate decision-making.

---

## Problem Statement

Property prices depend on multiple factors, making manual estimation difficult and often inconsistent.

The goal of this project is to:

* Predict house prices accurately.
* Analyze factors affecting property value.
* Compare multiple regression algorithms.
* Select the best-performing model for price estimation.

---

## Dataset Information

The dataset contains various housing-related attributes used for predicting house prices.

### Features Included

| Feature          | Description                   |
| ---------------- | ----------------------------- |
| Area             | Total area of the house       |
| Bedrooms         | Number of bedrooms            |
| Bathrooms        | Number of bathrooms           |
| Stories          | Number of floors              |
| Mainroad         | Access to main road           |
| Guestroom        | Availability of guest room    |
| Basement         | Availability of basement      |
| Hotwaterheating  | Hot water heating facility    |
| Airconditioning  | Air conditioning availability |
| Parking          | Number of parking spaces      |
| Prefarea         | Preferred area status         |
| Furnishingstatus | Furnishing condition          |
| Price            | Target Variable               |

### Target Variable

| Variable | Description      |
| -------- | ---------------- |
| Price    | House Sale Price |

---

## Project Workflow

```text
Data Collection
       ↓
Data Understanding
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Categorical Encoding
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Price Prediction
```

---

## Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib

---

## Exploratory Data Analysis (EDA)

The following analyses were performed:

### Data Inspection

* Dataset shape analysis
* Missing value verification
* Data type checking
* Statistical summary generation

### Visualizations

* House price distribution
* Area vs Price relationship
* Correlation heatmap
* Boxplots for outlier detection
* Feature-wise distribution analysis

### Key Insights

* Area has a strong positive correlation with house price.
* Houses with air conditioning and parking generally have higher prices.
* Preferred area properties tend to be more expensive.
* Furnishing status significantly affects property value.

---

## Data Preprocessing

### Data Cleaning

* Checked for missing values.
* Removed unnecessary columns if required.
* Verified data consistency.

### Categorical Encoding

Categorical features were transformed into numerical values using:

```python
LabelEncoder()
```
### Feature Scaling

Numerical variables were standardized when required.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Machine Learning Models Used

### 1. Linear Regression

* Baseline regression model
* Easy interpretation
* Fast training

### 2. Decision Tree Regressor

* Captures nonlinear relationships
* Easy visualization

### 3. Random Forest Regressor

* Ensemble learning model
* Better generalization
* Reduces overfitting

### 4. Gradient Boosting Regressor

* High predictive performance
* Handles complex relationships effectively

---

## Model Evaluation Metrics

The following metrics were used:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## Results

### Best Performing Model

🏆 **Linear Regressor** 

Reasons:

* Higher prediction accuracy.
* Better handling of nonlinear relationships.
* Reduced overfitting compared to individual decision trees.

---

## Important Features Affecting House Price

The model identified the following influential factors:

* Area
* Number of Bedrooms
* Number of Bathrooms
* Parking Availability
* Air Conditioning
* Preferred Area
* Furnishing Status

These features significantly contribute to house price estimation.

---

## Business Impact

This solution can help:

* Real Estate Agencies
* Property Buyers
* Property Sellers
* Housing Consultants
* Financial Institutions

Benefits include:

* Faster property valuation.
* Better investment decisions.
* Improved market analysis.
* Reduced manual appraisal efforts.

---

## Challenges Faced

* Handling categorical housing features.
* Managing outliers in property prices.
* Selecting the most suitable regression algorithm.
* Improving prediction accuracy.

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV.
* Advanced ensemble models such as XGBoost.
* Geographic location integration.
* Interactive price prediction dashboard.
* Cloud deployment.

---

## Repository Structure

```text
House-Price-Prediction/
│
├── data/
│   └── Housing.csv
│
├── notebooks/
│   └── House_Price_Prediction.ipynb
│
├── models/
│   └── house_price_model.pkl
│
├── README.md
├── requirements.txt
└── assets/
```

---

## Sample Input

| Feature          | Value     |
| ---------------- | --------- |
| Area             | 7420      |
| Bedrooms         | 4         |
| Bathrooms        | 2         |
| Stories          | 3         |
| Mainroad         | Yes       |
| Guestroom        | No        |
| Basement         | No        |
| Hotwaterheating  | No        |
| Airconditioning  | Yes       |
| Parking          | 2         |
| Prefarea         | Yes       |
| Furnishingstatus | Furnished |

---

## Prediction Output

**Predicted House Price:** ₹12,500,000

---

## Author

**Purandhar**

Machine Learning Project – House Price Prediction System
