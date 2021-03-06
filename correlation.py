import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv('input/train.csv')

# Convert string attributes to categorical attributes
df['Primary'] = df['Primary'].astype('category').cat.codes
df['Secondary'] = df['Secondary'].astype('category').cat.codes
df['Legendary'] = df['Legendary'].astype('category').cat.codes
print(df.info())

# Correlation Matrix
corr_matrix = df.corr()

corr_matrix.to_csv("out.csv")

print((corr_matrix))
sn.heatmap(corr_matrix, annot=True)
plt.show()
