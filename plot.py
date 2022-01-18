import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('test/train.csv')
test_data = pd.read_csv('test/test.csv')

print("train:")
print(train_data.describe())
print("===================")
print("test:")
print(test_data.describe())



'''


sns.displot(train_data['Capture_rate'])
plt.savefig('train.png')

sns.displot(test_data['Capture_rate'])
plt.savefig('test.png')
'''
