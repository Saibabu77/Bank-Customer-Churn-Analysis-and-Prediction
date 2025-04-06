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
```

### 2. Exploratory Data Analysis (EDA)

Plotting age distribution using Plotly:

```python
import plotly.express as px

age_counts = c_data['Customer_Age'].value_counts().sort_index().reset_index()
age_counts.columns = ['Customer_Age', 'Count']

fig = px.bar(age_counts, x='Customer_Age', y='Count', title='Customers by Age')
fig.show()
```

Card category and gender breakdown with multi-pie subplots:

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=2, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}],
                                           [{"type": "pie"}, {"type": "pie"}]])

# Overall gender distribution
fig.add_trace(go.Pie(labels=['Female', 'Male'],
                     values=c_data.Gender.value_counts()), row=1, col=1)

# Blue Card Holders by gender
blue = c_data.query('Card_Category == "Blue"')
fig.add_trace(go.Pie(labels=blue.Gender.value_counts().index,
                     values=blue.Gender.value_counts()), row=1, col=2)

fig.update_layout(title="Card Category by Gender")
fig.show()
```

### 3. Preprocessing

```python
c_data['Attrition_Flag'] = c_data['Attrition_Flag'].replace({'Attrited Customer': 1, 'Existing Customer': 0})
c_data['Gender'] = c_data['Gender'].replace({'F': 1, 'M': 0})

c_data = pd.concat([c_data, pd.get_dummies(c_data['Education_Level'], drop_first=True)], axis=1)
c_data.drop(['CLIENTNUM', 'Education_Level'], axis=1, inplace=True)
```

### 4. Class Imbalance Handling with SMOTE

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE()
X_res, y_res = sm.fit_resample(c_data.drop('Attrition_Flag', axis=1), c_data['Attrition_Flag'])
```

### 5. Dimensionality Reduction with PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=4)
pc_matrix = pca.fit_transform(X_res)
```

### 6. Model Training and Evaluation

```python
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X_res, y_res, cv=5, scoring='f1')
    print(f"{name} F1 Scores:", scores)
```



![image](https://github.com/user-attachments/assets/071045f2-f61f-40ff-9b5f-81c928a015cf)
This plot shows the 5-fold cross-validation F1 scores for three classification models: Random Forest, AdaBoost, and SVM.
Random Forest performed consistently well, with F1 scores slightly above 0.91 across all folds.
AdaBoost had stable performance around 0.87–0.88, with minor fluctuations.
SVM also maintained strong results, slightly under Random Forest, but above 0.87 overall.
Overall, Random Forest outperformed the others in terms of both stability and peak F1 score.




























## Tools Used

- Python (Pandas, Scikit-learn, Imbalanced-learn)
- Plotly, Seaborn, Matplotlib
- SMOTE for oversampling
- PCA for dimensionality reduction
- Classification Models: Random Forest, AdaBoost, SVM

## How to Run

1. Clone this repository  
2. Place `BankChurners.csv` in the root folder  
3. Install dependencies (Plotly, Scikit-learn, imbalanced-learn)  
4. Run the notebook file or use `jupyter nbconvert` to execute it end-to-end  
