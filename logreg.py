import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
seed = 11
df = pd.read_csv('banknote.csv')
df.reindex(np.random.permutation(df.index))
y = df['class']
X = df.ix[:,:-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1,random_state=seed)
logreg = LogisticRegression(solver='liblinear',tol=1e-1,C=1.e4 / X.shape[0])
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print "acc:",accuracy_score(y_test,y_pred)
