# Autism Spectrum Disorder (ASD) Detection Using Machine Learning

This project is focused on building a predictive model to detect the likelihood of Autism Spectrum Disorder (ASD) in individuals based on behavioral and personal data. It involves data preprocessing, exploratory data analysis (EDA), feature engineering, and training of machine learning models for classification.

## 📌 Project Objectives

- Load and preprocess ASD-related medical data.
- Perform exploratory data analysis to identify meaningful patterns.
- Apply machine learning techniques to classify individuals as ASD positive or negative.
- Evaluate the model's performance using appropriate classification metrics.

## 🧾 Dataset Description

The dataset contains various features including:

- Behavioral attributes (e.g., A1 to A10 survey responses)
- Demographic data (e.g., age, gender, ethnicity, country of residence)
- Screening scores
- Final diagnosis: `Class/ASD` (target variable)

The dataset was cleaned by:
- Removing irrelevant features like `age`, `id`
- Replacing categorical values with appropriate labels
- Converting float columns to integer where applicable
- Handling missing values in categorical and numerical data

## 🔍 Exploratory Data Analysis (EDA)

- Distribution plots for numerical and categorical features
- Value counts for categorical variables
- Correlation matrix and heatmap
- Outlier detection using IQR method
- Feature scaling using `StandardScaler`

## ⚙️ Model Building

The following machine learning models were tested:

- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**

### Feature Selection:
- Important features were selected using correlation analysis and domain relevance.

### Train-Test Split:
- Dataset split into training and testing sets using `train_test_split`.

## ✅ Model Evaluation

Models were evaluated using the following metrics:
- **Accuracy**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-score)**

Cross-validation and hyperparameter tuning were also performed to enhance model performance.

## 🔬 Results

- Random Forest and SVM showed the highest accuracy and robustness.
- Confusion matrices and classification reports provided insights into false positives and negatives.
- The final model was able to classify ASD with high precision and recall.

## 📊 Visualizations

- Heatmaps of correlations
- Box plots for outlier detection
- Bar plots and count plots for categorical analysis

## 🚀 How to Run

1. Clone the repository
2. Install the required libraries:

```bash
pip install -r requirements.txt
