This project involves an end-to-end analysis and modeling pipeline on a bank’s customer churn dataset. It includes exploratory data analysis (EDA), preprocessing, class balancing, dimensionality reduction, and supervised learning models.

Pfrom pathlib import Path

readme_text = """
# Bank Customer Churn Analysis and Prediction

This repository presents a detailed end-to-end analysis and machine learning pipeline focused on predicting customer churn in a banking environment.

## Dataset

- **File:** `BankChurners.csv`  
- **Target:** `Attrition_Flag`  
- **Objective:** Predict whether a customer will churn based on demographic and account activity data.

---

## Project Overview

### 1. Data Loading and Cleaning

```python
import pandas as pd

c_data = pd.read_csv('BankChurners.csv')
# Dropping unnecessary columns
c_data = c_data[c_data.columns[:-2]]
