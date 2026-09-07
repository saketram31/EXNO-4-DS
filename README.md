# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("bmi.csv") 
df.head()
```
<img width="516" height="266" alt="image" src="https://github.com/user-attachments/assets/0c6de546-39fa-4cd3-9752-134fd608ee35" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="396" height="141" alt="image" src="https://github.com/user-attachments/assets/a285a8e6-d884-4009-af90-17f7c51daa02" />

```
df.dropna()
```
<img width="671" height="527" alt="image" src="https://github.com/user-attachments/assets/17b94c3b-9c56-40ab-884a-c1382b932feb" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0) 
max_vals
```
<img width="342" height="98" alt="image" src="https://github.com/user-attachments/assets/21691c34-0dd0-432c-ab47-a5cec8b14ae2" />

```
from sklearn.preprocessing import StandardScaler 
df1=pd.read_csv("bmi.csv") 
df1.head()
```
<img width="591" height="267" alt="image" src="https://github.com/user-attachments/assets/12c66371-ac8f-4b04-a4be-d4a7cadd5bdb" />

```
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df1[['Height', 'Weight']] = sc.fit_transform(df1[['Height', 'Weight']])

df1.head(10)
```
<img width="558" height="457" alt="image" src="https://github.com/user-attachments/assets/8ed2b82b-7040-4b81-a96b-3e49f0a978c2" />

```
from sklearn.preprocessing import MinMaxScaler 
scaler=MinMaxScaler() 
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="591" height="436" alt="image" src="https://github.com/user-attachments/assets/22a91a3e-6d72-4fc7-a857-bfc192dc599d" />

```
from sklearn.preprocessing import MaxAbsScaler 
scaler = MaxAbsScaler()
df3=pd.read_csv("bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="661" height="515" alt="image" src="https://github.com/user-attachments/assets/5205a82b-73b1-4378-8c33-38a81a1e9c64" />

```
from sklearn.preprocessing import RobustScaler 
scaler = RobustScaler() 
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']]) 
df3.head()
```
<img width="562" height="275" alt="image" src="https://github.com/user-attachments/assets/0201fc43-d671-451a-8da2-0ef77638ae3d" />

```
df=pd.read_csv("income(1) (1).csv") 
df.info()
```
<img width="595" height="487" alt="image" src="https://github.com/user-attachments/assets/d8fef93b-81fc-4c00-929a-cb4ef2597889" />

```
df_null_sum=df.isnull().sum() 
df_null_sum
```
<img width="522" height="357" alt="image" src="https://github.com/user-attachments/assets/72481f46-3675-4a17-b016-d6f9f33aafc0" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry'] 
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="942" height="405" alt="image" src="https://github.com/user-attachments/assets/fdee9c7d-f720-49b3-8fa8-0242325e10f0" />

```
df[categorical_columns] = df[categorical_columns].astype('category') 
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes) 
df[categorical_columns]
```
<img width="937" height="400" alt="image" src="https://github.com/user-attachments/assets/f262d597-75a6-4e2b-a3f1-5aae5c9db256" />

```
X = df.drop(columns=['SalStat']) 
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train)
```
<img width="470" height="100" alt="image" src="https://github.com/user-attachments/assets/4e9ca9c5-540a-4bc4-b3c2-daa8dc9c5671" />

```
y_pred = rf.predict(X_test)
df=pd.read_csv("income(1) (1).csv") 
df.info()
```

<img width="463" height="377" alt="image" src="https://github.com/user-attachments/assets/e6b65edb-a505-49ee-9ba4-8f0620ab16df" />

```
import pandas as pd 
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry'] 
df[categorical_columns] = df[categorical_columns].astype('category') 
df[categorical_columns]
```
<img width="936" height="407" alt="image" src="https://github.com/user-attachments/assets/373999a6-4aee-4ec8-b4ae-a7306292c950" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="807" height="406" alt="image" src="https://github.com/user-attachments/assets/eeaf6923-8b23-4321-a812-7e959c748d6b" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat'] 
k_chi2 = 6 
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y) 
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
<img width="683" height="87" alt="image" src="https://github.com/user-attachments/assets/d9bb0659-270f-4b0f-94bd-f8bac81c8bc5" />

```
import pandas as pd 
from sklearn.feature_selection import SelectKBest, chi2, f_classif 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss', 'hoursperweek']
X = df[selected_features] 
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train)
```
<img width="453" height="91" alt="image" src="https://github.com/user-attachments/assets/57f0b329-de28-43d8-a002-4ca75ee15665" />

```
y_pred = rf.predict(X_test) 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
<img width="557" height="41" alt="image" src="https://github.com/user-attachments/assets/05ae33ff-92d0-4a52-82c8-4b1716e622c2" />

```
!pip install skfeature-chappers
```
<img width="1766" height="485" alt="500892749-b3de7d8a-3bcd-450f-b054-7ed0d5c879af" src="https://github.com/user-attachments/assets/a67e6f4a-3cee-40a6-9a59-70937e862015" />

```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
```
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
<img width="1359" height="780" alt="500893464-1c020df4-6f23-43f1-a564-742813cafc50" src="https://github.com/user-attachments/assets/07114a0e-c50d-4c73-994f-81062752284a" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
```
```
selected_features_anova = X.columns[selector_anova.get_support()]
```
```
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
<img width="1329" height="191" alt="500894042-3b3ddcff-1347-45bd-a0de-c682721a6812" src="https://github.com/user-attachments/assets/1e4076ee-08ca-40d9-b401-2dda1422df2d" />

```
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
```
```
df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```
```
df[categorical_columns]
```
<img width="1496" height="679" alt="500895176-2030b6ef-7828-406d-93d2-097f8ac96beb" src="https://github.com/user-attachments/assets/79a14193-a5e9-48bf-ae63-35c565f1e1f0" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
logreg = LogisticRegression()
```
```
n_features_to_select =6
```
```
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
<img width="1655" height="702" alt="500896102-7ed7cf84-dcb9-4ce3-9d26-029b508b519d" src="https://github.com/user-attachments/assets/3e15d6e8-c2b8-455a-bed3-ac2495e46f80" />

<img width="1777" height="793" alt="500896260-d49f5b8d-7ddc-46be-87d7-bb4601c68f16" src="https://github.com/user-attachments/assets/65cd12ed-74d3-4da5-9828-67023bdb513f" />



# RESULT:
  Given code is executed successfully...
