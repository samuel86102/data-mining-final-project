from mylib import discretize
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('input/train.csv')

features = ['Attack','Defense','Escape_rate','MaxCP']

print(df['Capture_rate'].describe())
discretize(df)
print(df['Capture_rate'].describe())



X = df[features]

y = df['Capture_rate']



