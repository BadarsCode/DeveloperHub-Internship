# ❤️ Heart Disease Prediction System

## 📌 Project Overview
This project focuses on predicting whether a person is at risk of heart disease based on medical attributes using a **Random Forest classification model**. The system is designed to assist in early risk assessment using machine learning.

---

## 🎯 Objective
To build a binary classification model that predicts the presence or absence of heart disease using patient health data.

---

## 📊 Dataset
- **Name:** Heart Disease UCI Dataset
- **Source:** Kaggle
- **Target Variable:** `target`
  - `1` → Heart disease present
  - `0` → No heart disease

### 🔢 Features Used
- age
- sex
- cp (chest pain type)
- trestbps (resting blood pressure)
- chol (serum cholesterol)
- fbs (fasting blood sugar)
- restecg (resting ECG)
- thalach (max heart rate)
- exang (exercise induced angina)
- oldpeak (ST depression)
- slope
- ca (major vessels)
- thal

---

## 🧹 Data Preprocessing
- Checked for missing values
- Handled numerical consistency
- Selected relevant medical features
- No scaling required (Random Forest is scale-independent)

---

## 🤖 Model Used
- **Algorithm:** Random Forest Classifier
- **Reason:** Handles non-linear medical data well and provides feature importance

---

## 📈 Model Evaluation
- Accuracy
- Confusion Matrix
- ROC Curve & AUC Score
- Feature Importance Analysis

---

## 🧠 Key Findings
- Chest pain type, ST depression, and number of vessels are strong predictors
- Model generalizes well on unseen data
- Suitable for medical risk screening (not diagnosis)

---

## 🖥 Web Application
- Built using **Streamlit**
- Allows user input of medical parameters
- Displays:
  - Risk prediction
  - Probability score
  - Confusion matrix heatmap
  - ROC curve
  - Feature importance chart

---

## 🛠 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

---

## ⚠ Disclaimer
This project is for educational purposes only and should not be used as a medical diagnosis tool.

---

## 📌 Author
Subhanud Din
