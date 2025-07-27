import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split

#labelencoder - to convert text categories into numerical categories
#Standardscalar - to standardize numerical data

#load dataset
df=pd.read_csv('data/telco_customer_churn.csv')

#Handle Missing Value
df['TotalCharges']=pd.to_numeric(df['TotalCharges'] , errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean() , inplace =True)

le =LabelEncoder()
categorical_cols=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod', 'Churn']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

#Define features and target
X= df.drop(['customerID','Churn'],axis=1)
y=df['Churn']

#scale numerical features
scaler=StandardScaler()
numerical_cols = ['tenure','MonthlyCharges','TotalCharges']
X[numerical_cols]= scaler.fit_transform(X[numerical_cols])

#Split dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)