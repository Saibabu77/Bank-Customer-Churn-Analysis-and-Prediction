This project involves an end-to-end analysis and modeling pipeline on a bank’s customer churn dataset. It includes exploratory data analysis (EDA), preprocessing, class balancing, dimensionality reduction, and supervised learning models.

2. Exploratory Data Analysis (EDA)
Plotting age distribution using Plotly:

python
Always show details

Copy
import plotly.express as px

age_counts = c_data['Customer_Age'].value_counts().sort_index().reset_index()
age_counts.columns = ['Customer_Age', 'Count']

fig = px.bar(age_counts, x='Customer_Age', y='Count', title='Customers by Age')
fig.show()
Card category and gender breakdown with multi-pie subplots:

python
Always show details

Copy
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
3. Preprocessing
python
Always show details

Copy
c_data['Attrition_Flag'] = c_data['Attrition_Flag'].replace({'Attrited Customer': 1, 'Existing Customer': 0})
c_data['Gender'] = c_data['Gender'].replace({'F': 1, 'M': 0})

c_data = pd.concat([c_data, pd.get_dummies(c_data['Education_Level'], drop_first=True)], axis=1)
c_data.drop(['CLIENTNUM', 'Education_Level'], axis=1, inplace=True)
4. Class Imbalance Handling with SMOTE
python
Always show details

Copy
from imblearn.over_sampling import SMOTE

sm = SMOTE()
X_res, y_res = sm.fit_resample(c_data.drop('Attrition_Flag', axis=1), c_data['Attrition_Flag'])
5. Dimensionality Reduction with PCA
python
Always show details

Copy
from sklearn.decomposition import PCA

pca = PCA(n_components=4)
pc_matrix = pca.fit_transform(X_res)
6. Model Training and Evaluation
python
Always show details

Copy
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
Tools Used
Python (Pandas, Scikit-learn, Imbalanced-learn)

Plotly, Seaborn, Matplotlib

SMOTE for oversampling

PCA for dimensionality reduction

Classification Models: Random Forest, AdaBoost, SVM
