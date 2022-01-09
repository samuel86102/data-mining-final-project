# Data Mining Final Project - 07
## Pokemon Dataset
![](img/logo.png)
Source:[https://www.kaggle.com/netzuel/pokmon-go-dataset-15-generations](https://www.kaggle.com/netzuel/pokmon-go-dataset-15-generations)

## Data Preprocessing
決定取用的feature前，先身對features進行correlation的分析，由於`Primary`、`Secondary`和`Lengendary`的類型皆為string，因此使用以下function轉換為數字的表示法:

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
### Correlation Matrix

```
               Pokedex   Stamina    Attack   Defense   Primary  Secondary  \
Pokedex       1.000000  0.120524  0.175620  0.098728 -0.082739  -0.113946   
Stamina       0.120524  1.000000  0.409917  0.348467  0.137479  -0.068558   
Attack        0.175620  0.409917  1.000000  0.487408 -0.052026  -0.078269   
Defense       0.098728  0.348467  0.487408  1.000000  0.083015   0.063235   
Primary      -0.082739  0.137479 -0.052026  0.083015  1.000000  -0.022269   
Secondary    -0.113946 -0.068558 -0.078269  0.063235 -0.022269   1.000000   
MaxHP         0.120753  0.999969  0.409392  0.348522  0.137427  -0.068337   
Capture_rate -0.133526 -0.364982 -0.501927 -0.484784  0.110998   0.070552   
Escape_rate  -0.146795 -0.361632 -0.410829 -0.482166 -0.001220   0.057452   
Weight        0.114777  0.386328  0.401872  0.472485  0.092456  -0.030935   
Height       -0.006375  0.391514  0.394973  0.405106  0.081867  -0.087255   
Legendary     0.141434  0.223790  0.332386  0.384258  0.005095  -0.080820   
MaxCP         0.185219  0.600582  0.923271  0.708356  0.013695  -0.056951   
Generation    0.977453  0.085094  0.122374  0.040476 -0.088519  -0.108362   

                 MaxHP  Capture_rate  Escape_rate    Weight    Height  \
Pokedex       0.120753     -0.133526    -0.146795  0.114777 -0.006375   
Stamina       0.999969     -0.364982    -0.361632  0.386328  0.391514   
Attack        0.409392     -0.501927    -0.410829  0.401872  0.394973   
Defense       0.348522     -0.484784    -0.482166  0.472485  0.405106   
Primary       0.137427      0.110998    -0.001220  0.092456  0.081867   
Secondary    -0.068337      0.070552     0.057452 -0.030935 -0.087255   
MaxHP         1.000000     -0.365300    -0.361923  0.386662  0.391624   
Capture_rate -0.365300      1.000000     0.499510 -0.362570 -0.355135   
Escape_rate  -0.361923      0.499510     1.000000 -0.269525 -0.255778   
Weight        0.386662     -0.362570    -0.269525  1.000000  0.637847   
Height        0.391624     -0.355135    -0.255778  0.637847  1.000000   
Legendary     0.224283     -0.211957    -0.261289  0.426227  0.272710   
MaxCP         0.600254     -0.569315    -0.506865  0.540116  0.487339   
Generation    0.085294     -0.119957    -0.117182  0.082506 -0.039373   

              Legendary     MaxCP  Generation  
Pokedex        0.141434  0.185219    0.977453  
Stamina        0.223790  0.600582    0.085094  
Attack         0.332386  0.923271    0.122374  
Defense        0.384258  0.708356    0.040476  
Primary        0.005095  0.013695   -0.088519  
Secondary     -0.080820 -0.056951   -0.108362  
MaxHP          0.224283  0.600254    0.085294  
Capture_rate  -0.211957 -0.569315   -0.119957  
Escape_rate   -0.261289 -0.506865   -0.117182  
Weight         0.426227  0.540116    0.082506  
Height         0.272710  0.487339   -0.039373  
Legendary      1.000000  0.445100    0.099637  
MaxCP          0.445100  1.000000    0.122806  
Generation     0.099637  0.122806    1.000000 
```
### Heatmap
![](img/corr.png)
