from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def discretize(df, feature):
    disc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
    disc.fit(df[[feature]])
    #print(disc.bin_edges_)
    train_t = disc.transform(df[[feature]])
    return pd.DataFrame(train_t, columns = [feature])

# Import Data
df = pd.read_csv('input/train.csv') 
df_test = pd.read_csv('input/test.csv')

# Preprocessing

## Equal Width Discretization
df['Capture_rate'] = discretize(df,'Capture_rate')
df_test['Capture_rate'] = discretize(df_test,'Capture_rate')


## Plot the distribution of 'Capture_rate' in training data and testing data
train_plt = sns.displot(df['Capture_rate'],color='#A0E7E5').set(title='Training Data')
plt.tight_layout()
plt.savefig('train.png')
test_plt = sns.displot(df_test['Capture_rate'],color='#FFAEBC').set(title='Testing Data')
plt.tight_layout()
plt.savefig('test.png')
plt.show()


## Drop unnecessary features and split data into training data and validation data
X = df.drop(['Name','Pokedex','Primary','Secondary','Legendary','Capture_rate'], axis='columns')
y = df['Capture_rate']
#train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)
train_X = X
train_y = y

## Build model
clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
clf.fit(train_X, train_y)
# Export tree structure
export_graphviz(clf, out_file='tree.dot')
'''
accuracy = clf.score(test_X, test_y)
print(accuracy)
'''
## Testing data
X_ = df_test.drop(['Name','Pokedex','Primary','Secondary','Legendary','Capture_rate'],axis='columns')
y_ = df_test['Capture_rate']
accuracy_test = clf.score(X_, y_)
print(accuracy_test)


