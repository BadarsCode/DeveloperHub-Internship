News Topic Classifier using BERT

An end-to-end NLP project that fine-tunes a Transformer-based model (BERT) to classify news headlines into topic categories and deploys it as an interactive web application using Streamlit.

Project Overview

This project builds a News Topic Classification system using a pre-trained BERT model fine-tuned on the AG News dataset.

The model classifies news headlines into four categories:

World

Sports

Business

Sci/Tech

The trained model is deployed using Streamlit, allowing real-time interaction through a web interface.

Dataset

Dataset Name: AG News Dataset

Source: Hugging Face Datasets

Total Samples:

120,000 training samples

7,600 test samples

Classes: 4 topic categories

Each sample contains:

text → News headline

label → Topic category (0–3)

Technologies Used

Transformers (Hugging Face) – Fine-tuned bert-base-uncased

PyTorch – Deep learning framework

Hugging Face Datasets – Data loading & preprocessing

Scikit-learn / Evaluate – Accuracy & F1-score metrics

Streamlit – Web app deployment

Python 3.10+

Project Structure
News-Topic-Classifier/
│
├── news-bert-model/        # Saved fine-tuned model
├── streamlit_app.py        # Streamlit deployment script
├── train.py                # Training script
├── requirements.txt
└── README.md

Model Training Process

Loaded AG News dataset using Hugging Face

Tokenized text using BERT tokenizer

Fine-tuned bert-base-uncased

Evaluated model using:

Accuracy

Weighted F1-score

Saved trained model for inference

Model Performance

After fine-tuning:

Accuracy: ~93–95%

Weighted F1-score: ~93–95%

(The exact score may vary depending on epochs and hyperparameters.)

Streamlit Deployment

The trained model is deployed using Streamlit for real-time interaction.

Features:

User inputs a news headline

Model predicts category

Displays probability for each class

Instant inference

How to Run Locally
Clone the repository
git clone https://github.com/BadarsCode/DeveloperHub-Internship.git
cd news-topic-classifier

Install dependencies
pip install -r requirements.txt

Run the Streamlit app
streamlit run streamlit_app.py


The app will open in your browser.

Requirements

Example requirements.txt:

torch
transformers
datasets
evaluate
scikit-learn
streamlit

Key Learnings

Fine-tuning Transformer models for NLP tasks

Handling tokenization and padding correctly

Using Trainer API for efficient training

Evaluating classification models properly

Deploying ML models into interactive web applications

Future Improvements

Deploy to Hugging Face Spaces / Streamlit Cloud

Add confusion matrix visualization

Implement batch prediction

Add confidence threshold filtering

Convert to FastAPI backend

Author

Subhanud Din
Machine Learning & AI Enthusiast