import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('loan_data.csv')
df.drop("ACC_NO", axis=1, inplace=True)
cols = ["INF_MARITAL_STATUS", "INF_GENDER", "COMPENSATION_CHARGED"]

for col in cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

df["CLIENT_TYPE"].fillna(df["CLIENT_TYPE"].mode()[0], inplace=True)
df["INSTALL_SIZE"].fillna(df["INSTALL_SIZE"].median(), inplace=True)
df.drop_duplicates(inplace=True)
df['COMPENSATION_CHARGED'] = pd.get_dummies(df["COMPENSATION_CHARGED"],dtype=int,drop_first=True) #also we can give boolean
df['REPAY_MODE'] = pd.get_dummies(df['REPAY_MODE'],dtype=int,drop_first=True)

df = df[df["QUALITY_OF_LOAN"].isin(['G', 'B'])]   # adjust based on actual labels




from sklearn.preprocessing import LabelEncoder

encoders = {}

# Marital Status
le_marital = LabelEncoder()
df['INF_MARITAL_STATUS'] = le_marital.fit_transform(df['INF_MARITAL_STATUS'])
encoders['INF_MARITAL_STATUS'] = le_marital

# Gender
le_gender = LabelEncoder()
df['INF_GENDER'] = le_gender.fit_transform(df['INF_GENDER'])
encoders['INF_GENDER'] = le_gender

# Client Type
df["CLIENT_TYPE"] = df["CLIENT_TYPE"].astype(str)
le_client = LabelEncoder()
df['CLIENT_TYPE'] = le_client.fit_transform(df['CLIENT_TYPE'])
encoders['CLIENT_TYPE'] = le_client

# Target
le_target = LabelEncoder()
df['QUALITY_OF_LOAN'] = le_target.fit_transform(df['QUALITY_OF_LOAN'])
encoders['QUALITY_OF_LOAN'] = le_target

x=df.drop('QUALITY_OF_LOAN',axis=1)
y=df['QUALITY_OF_LOAN']

feature_columns = x.columns.tolist()

from sklearn.preprocessing import MinMaxScaler,StandardScaler
normalisation = MinMaxScaler()
x_scaled=normalisation.fit_transform(x)

x=pd.DataFrame(x_scaled)
with open('scaling.pkl', 'wb') as f:
    pickle.dump(normalisation, f)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42, stratify=y
)

from sklearn.ensemble import RandomForestClassifier

model  = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model .fit(x_train, y_train)

import os
print("Current folder:", os.getcwd())

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Step 5: model saved as model.pkl")


with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)


with open("feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

