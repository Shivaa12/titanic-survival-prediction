# titanic-survival-prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('C:/Users/shiva/OneDrive/Desktop/internship encryptix/titanic/Titanic-Dataset.csv')
test = pd.read_csv('C:/Users/shiva/OneDrive/Desktop/internship encryptix/titanic/Titanic-Dataset.csv')

print(train.head())
print(train.info())
print(train.describe())

train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

le = LabelEncoder()
train['Embarked'] = le.fit_transform(train['Embarked'])
test['Embarked'] = le.transform(test['Embarked'])
train.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
test.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
X_train, X_val, y_train, y_val = train_test_split(train.drop('Survived', axis=1), train['Survived'], test_size=0.2, random_state=42)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=['Not Survived', 'Survived']))
print("Confusion Matrix:")
print(pd.crosstab(y_val, y_pred, rownames=['True'], colnames=['Predicted']))
test_pred = rfc.predict(test.drop('Survived', axis=1))
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': test_pred})
submission.to_csv('submission.csv', index=False)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': test_pred})
submission.to_csv('submission.csv', index=False)
