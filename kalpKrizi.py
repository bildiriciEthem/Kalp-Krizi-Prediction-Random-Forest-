import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

veriler=pd.read_csv('heart.csv')
korelasyon=veriler.corr()
x=veriler.iloc[:,0:13].values
y=veriler.iloc[:,13:14].values

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.10,random_state=0)


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

rfc=RandomForestClassifier(n_estimators=100,max_depth=5)
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
y_pred=(y_pred>0.5)

cm=confusion_matrix(y_test, y_pred)
print("Başarı oranı:")
print(100*accuracy_score(y_test,y_pred))
print("Confusion_matrix:")
print(cm)



