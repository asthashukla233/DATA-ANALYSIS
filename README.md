from google.colab import drive
drive.mount('/content/path')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
path=('/content/path/MyDrive/csv files/customer_churn_data.csv')
df=pd.read_csv(path)
df.head()
df.shape
df.info()
df.isnull().sum()
mode=df['InternetService'].mode()[0]
df['InternetService'].fillna(mode)
df['Gender'].unique(),df['ContractType'].unique(),df['InternetService'].unique(),df['TechSupport'].unique(),df['Churn'].unique()
for col in ['Gender', 'ContractType', 'InternetService', 'TechSupport', 'Churn']:
    df[col] = df[col].astype(str)
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.hist(df['Gender'])
plt.title('Gender Distribution')
plt.subplot(2, 3, 2)
plt.hist(df['ContractType'])
plt.title('Contract Type Distribution')
plt.subplot(2, 3, 3)
plt.hist(df['InternetService'])
plt.title('Internet Service Distribution')
plt.subplot(2, 3, 4)
plt.hist(df['TechSupport'])
plt.title('Tech Support Distribution')
plt.subplot(2, 3, 5)
plt.hist(df['Churn'])
plt.title('Churn Distribution')
plt.tight_layout()
sns.barplot(x='Gender', y='Age',hue='InternetService' ,data=df)
sns.scatterplot(x='Age', y='MonthlyCharges', hue='Churn', data=df)
sns.violinplot(x='Churn', y='Age', data=df)
sns.boxplot(x='InternetService', y='Age', data=df)
sns.boxplot(x='Churn', y='Age', data=df)
sns.countplot(x='InternetService', hue='Churn', data=df)
sns.countplot(x='ContractType', hue='Churn', data=df)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
categorical_features = ['Gender', 'ContractType', 'InternetService', 'TechSupport']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
categorical_features = ['Gender', 'ContractType', 'InternetService', 'TechSupport']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Predicted Values:")
print(y_pred)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
categorical_features = ['Gender', 'ContractType', 'InternetService', 'TechSupport']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
results_df = pd.DataFrame({'Actual Churn': y_test, 'Predicted Churn': y_pred})
results_df = pd.concat([X_test, results_df], axis=1)print(results_df)









