End-to-End Customer Churn Prediction Pipeline

A production-ready Machine Learning pipeline built using Scikit-learn Pipeline API to predict customer churn using the Telco Customer Churn dataset.

This project demonstrates how to build reusable ML workflows with preprocessing, hyperparameter tuning, and model export for deployment.

Project Overview

Customer churn prediction is a critical business problem in the telecom industry. This project builds a complete ML pipeline that:

Preprocesses numerical and categorical data

Trains multiple models

Performs hyperparameter tuning using GridSearchCV

Exports a production-ready model

Ensures reusability and scalability

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Joblib

Dataset

Telco Customer Churn Dataset

Features include:

Customer demographics

Account information

Services subscribed

Monthly charges

Contract type

Tenure

Payment method

Target:

Churn (Yes / No)

ML Pipeline Architecture

The project uses Scikit-learn’s:

Pipeline

ColumnTransformer

GridSearchCV

Preprocessing Steps

Numerical Features

Missing value imputation (Median)

Standard Scaling

Categorical Features

Missing value imputation (Most Frequent)

OneHot Encoding (handle_unknown="ignore")

Models Trained
Logistic Regression

Tuned using GridSearchCV

Regularization parameter optimization

Random Forest

Tuned for:

n_estimators

max_depth

min_samples_split

Model Evaluation

Models are evaluated using:

Accuracy Score

Precision

Recall

F1-Score

Cross-validation (5-fold)

Model Export

The best-performing pipeline is saved using:

joblib.dump(model, "churn_prediction_pipeline.pkl")


This ensures:

Reusability

Easy deployment

Consistent preprocessing during inference

How to Use the Saved Model
import joblib

model = joblib.load("churn_prediction_pipeline.pkl")
prediction = model.predict(new_data)

📁 Project Structure
├── Telco-Customer-Churn.csv
├── churn_pipeline.ipynb
├── churn_prediction_pipeline.pkl
├── requirements.txt
└── README.md

🧪 How to Run
Clone the repository
git clone https://github.com/BadarsCode/DeveloperHub-Internship.git
cd your-repo-name

Install dependencies
pip install -r requirements.txt

Run the notebook

Open Jupyter Notebook:

jupyter notebook

Skills Demonstrated

ML Pipeline construction

Feature preprocessing using ColumnTransformer

Hyperparameter tuning with GridSearchCV

Model evaluation & comparison

Model serialization with Joblib

Production-ready ML workflow design

Key Learnings

Why pipelines are essential for real-world ML

Avoiding data leakage

Building reproducible ML systems

Preparing models for deployment

Future Improvements

Add SMOTE for class imbalance handling

Add ROC-AUC evaluation

Deploy using Streamlit

Serve using FastAPI

Add CI/CD for ML workflow

Author

Subhanud Din

Machine Learning & AI Enthusiast
Building production-ready ML systems 🚀