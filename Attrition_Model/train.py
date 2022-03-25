import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

url="https://raw.githubusercontent.com/PoojanSheth28/Attrition_Model/main/csv_file/HR_Employee_Attrition_Data.csv"
att_df=pd.read_csv(url)
print(att_df.head())

emp_df=att_df.drop(columns=['EmployeeCount','Over18','StandardHours','EmployeeNumber'],axis=1)

new_emp_df=emp_df.copy()
lst=[]
lst.append(new_emp_df.select_dtypes('int64').columns)

def iqr_capping(df,col,factor):
    for i in col:
        q1=df[i].quantile(0.25)
        q3=df[i].quantile(0.75)
        iqr=q3-q1
        upper=q3+(factor*iqr)
        lower=q1-(factor*iqr)
        df[i]=np.where(df[i]>upper,upper,np.where(df[i]<lower,lower,df[i]))
    return

iqr_capping(new_emp_df,lst,1.5)

hr_emp_df=new_emp_df.drop(columns=['PerformanceRating','JobLevel'],axis=1)


df=hr_emp_df.copy()
# cat_col=[]
# cat_col.append(str(df.select_dtypes('object').columns))

le = LabelEncoder()

mapping_dict ={}
for col in df.select_dtypes('object').columns:
    
    df[col] = le.fit_transform(df[col])
 
    le_name_mapping = dict(zip(le.classes_,
                        le.transform(le.classes_)))
 
    mapping_dict[col]= le_name_mapping
print(mapping_dict)
joblib.dump(mapping_dict,"mapping_dict.pkl")

x = df.iloc[:, df.columns != 'Attrition']
y = df.iloc[:, df.columns == 'Attrition']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2)

model = RandomForestClassifier(random_state=42)

model.set_params(criterion = 'gini',
                  max_features = 'auto', 
                  max_depth = 20,
                  min_samples_split=5,
                 min_samples_leaf=2,
                 n_estimators=1700,
                  bootstrap = True)

model.fit(x_train, y_train)

accuracy_model = model.score(x_test, y_test)
print("Mean accuracy on the test set is: ",accuracy_model*100,'%')
print(df.columns)

joblib.dump(model,"attrition_95.pkl")