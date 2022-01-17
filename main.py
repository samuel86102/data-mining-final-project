import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from mylib import discretize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv('input/train.csv')

features = ['Attack','Defense','Escape_rate','MaxCP']

discretize(df)

X = df[features]
y = df['Capture_rate']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)


clf = DecisionTreeClassifier()
clf.fit(train_X, train_y)
predicted = clf.predict(test_X)

# Accuracy
accuracy = accuracy_score(test_y, predicted)
print(accuracy)

# Confusion Matrix
cmatrix = pd.DataFrame(
        confusion_matrix(test_y, predicted),
        columns=['Predicted 0', 'Predicted 1'],
        index=['Actual 0', 'Actual 1'])
print(cmatrix)

# classification report
print(classification_report(test_y,pred))


