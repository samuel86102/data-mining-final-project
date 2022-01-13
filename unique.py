import pandas as pd
df = pd.read_csv('input/train.csv')

cr_num = (df["Capture_rate"].unique())
er_num = (df["Escape_rate"].unique())

print(str(len(cr_num))+" unique Values in Capture_rate:"+str(cr_num))
print(str(len(er_num))+" unique Values in Escape_rate:"+str(er_num))



