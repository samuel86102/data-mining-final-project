# Data Mining Final Project - 07
## Pokemon Dataset
![](img/logo.png)
dataset:[https://www.kaggle.com/netzuel/pokmon-go-dataset-15-generations](https://www.kaggle.com/netzuel/pokmon-go-dataset-15-generations)

## Data Preprocessing
決定取用的feature前，先身對features進行correlation的分析，油魚`Primary`、`Secondary`和`Lengendary`的類型皆為string，因此使用以下function轉換為數字的表示法:
```python
# Convert string attributes to categorical attributes
df['Primary'] = df['Primary'].astype('category').cat.codes
df['Secondary'] = df['Secondary'].astype('category').cat.codes
df['Legendary'] = df['Legendary'].astype('category').cat.codes
```

接著使用seanborn的heatmap畫出correlation matrix:
```python
corr_matrix = df.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()
```

![](img/corr.png)









