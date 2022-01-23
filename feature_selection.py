from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def discretize(df, feature):
    disc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
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


data = []
features = ['Stamina','Attack','Defense','MaxHP','Escape_rate','Weight','Height','MaxCP','Generation']
selected_features = ['Attack','Defense','MaxCP','Escape_rate']


y = df['Capture_rate']
#X = df.drop(['Name','Pokedex','Primary','Secondary','Legendary','Capture_rate'], axis='columns')
X = df.drop(['Attack',"MaxCP",'Name','Pokedex','Primary','Secondary','Legendary','Capture_rate'], axis='columns')

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 1)
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1)
clf.fit(train_X, train_y)
accuracy = clf.score(test_X, test_y)
data.append(accuracy)
print(accuracy)
'''
X = df[selected_features]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 1)
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1)
clf.fit(train_X, train_y)
accuracy = clf.score(test_X, test_y)
data.append(accuracy)
print(accuracy)

plt.ylabel('accuracy')
plt.bar(['9-features','4-features'],data)
plt.savefig('9vs4.png')
'''
'''
for e in features:
    X = df.drop(e, axis='columns')
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 1)
    clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1)
    clf.fit(train_X, train_y)
    accuracy = clf.score(test_X, test_y)
    data.append(accuracy)
    print(e)
    print(accuracy)
plt.ylabel('accuracy')
plt.xticks(rotation=45)
plt.bar(features,data)
plt.tight_layout()
plt.savefig('multi.png')
'''
