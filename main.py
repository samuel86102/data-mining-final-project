import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from mylib import discretize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import KBinsDiscretizer, Normalizer
from sklearn.tree import export_graphviz

df = pd.read_csv('input/train.csv')
df_test = pd.read_csv('input/test.csv')

#features = ['Attack','Defense','Escape_rate','MaxCP']

#df = discretize(df)

# Equal Frequency/Width Distretization

disc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
disc.fit(df[['Capture_rate']])
train_t = disc.transform(df[['Capture_rate']])
df['Capture_rate'] = pd.DataFrame(train_t, columns = ['Capture_rate'])

disc.fit(df_test[['Capture_rate']])
train_t = disc.transform(df_test[['Capture_rate']])
df_test['Capture_rate'] = pd.DataFrame(train_t, columns = ['Capture_rate'])

# Encode Legendary as numbers
df['Legendary'] = df['Legendary'].astype('category').cat.codes
df_test['Legendary'] = df_test['Legendary'].astype('category').cat.codes

'''
sns.displot(df['Capture_rate'])
sns.displot(df_test['Capture_rate'])
plt.show()
'''
print(disc.bin_edges_)

#print(df.Capture_rate.value_counts())
#sns.countplot(x='Capture_rate',data=df)
#plt.show()

#X = df[features]


X = df.drop(['Name','Pokedex','Primary','Secondary','Capture_rate'],axis='columns')
y = df['Capture_rate']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)
clf = DecisionTreeClassifier()
clf.fit(train_X, train_y)
#print(clf.score(test_X,test_y))
X_ = df.drop(['Name','Pokedex','Primary','Secondary','Capture_rate'],axis='columns')
y_ = df['Capture_rate']
print(clf.score(X_,y_))

#predicted = clf.predict(test_X)
'''
cri = ['gini','entropy']
for e in cri:
    clf = DecisionTreeClassifier(criterion=e)
    clf.fit(train_X, train_y)
    print(clf.score(test_X,test_y))
    predicted = clf.predict(test_X)

'''
'''
export_graphviz(clf, out_file='tree.dot',
                feature_names=['Attack', 'Defense','Escape_rate','MaxCP'])
'''
# Accuracy
#accuracy = accuracy_score(test_y, predicted)
#print(accuracy)
# classification report
# print(classification_report(test_y,predicted))

