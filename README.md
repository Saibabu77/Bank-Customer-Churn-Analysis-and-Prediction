This project involves an end-to-end analysis and modeling pipeline on a bank’s customer churn dataset. It includes exploratory data analysis (EDA), preprocessing, class balancing, dimensionality reduction, and supervised learning models.

Project Overview
1. Data Loading and Cleaning
python
Copy
Edit
import pandas as pd

c_data = pd.read_csv('BankChurners.csv')
# Dropping unnecessary columns
c_data = c_data[c_data.columns[:-2]]
