import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lib import svmBinary

df=pd.read_csv('data.csv',header=None)
#print(df.iloc[:,:-1])
#print(df.iloc[:,-1])

X=df.iloc[:,:-1]
X=X.values.tolist()
#print(X)
Y=df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print(X_train, y_train)
print(type(X_train))
input()
W=svmBinary.fit(X_train,y_train.values.tolist())
print(W)
print(X_test)
print(type(X_test))
input()
predicted,distance=svmBinary.predict(X_test,W)
print(predicted)
print(accuracy_score(y_test,predicted))

