# Customer Churn Prediction Project

## Overview
This project predicts customer churn for a telecommunications company using machine learning. It includes exploratory data analysis (EDA), data preprocessing, model building with Logistic Regression and Random Forest, and evaluation using metrics like accuracy, F1-score, and ROC-AUC. The goal is to identify customers likely to churn and provide insights for retention strategies.

## Dataset
The [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle is used, containing customer information such as tenure, contract type, monthly charges, and churn status.

## Project Structure
- `data/`: Contains the dataset and preprocessed files (not included in Git due to `.gitignore`).
- `notebooks/`: Jupyter notebooks for EDA (`eda.ipynb`).
- `scripts/`: Python scripts for preprocessing (`preprocessing.py`) and model training (`train_models.py`).
- `models/`: Saved machine learning models (not included in Git).
- `docs/`: Visualizations from EDA and model evaluation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/g0ku103/Churn-Prediction-Project.git

## Set up a virtual environment:
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

## Install dependencies:
pip install -r requirements.txt


## Result
Logistic Regression: 
Accuracy:0.8176
F1 Score : 0.6281
ROC-AUC : 0.8614

Random Forest: 
Accuracy: 0.7956
F1 Score: 0.5486
ROC-AUC: 0.8361