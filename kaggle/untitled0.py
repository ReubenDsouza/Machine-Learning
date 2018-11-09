import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import IsolationForest



#model=BaggingClassifier()
#model=AdaBoostClassifier()
model=XGBClassifier()
#model=ExtraTreesClassifier()
#model=RandomForestClassifier(min_samples_split=5)
#model=KNeighborsClassifier(n_neighbors=6)
#model = SVC(kernel='rbf')

X=data.drop(['ID', 'class'], axis=1)

Y=data['class']

from sklearn.model_selection import train_test_split
train_X, cross_val_X, train_Y, cross_val_Y = train_test_split(X, Y,random_state = 0)

#train_X -->train_Y
#cross_val_X-->cross_val_Y

model.fit(train_X,train_Y)

predicted_Y=model.predict(cross_val_X)

from sklearn.metrics import accuracy_score
print(accuracy_score(cross_val_Y,predicted_Y))

test_X=test.drop(['ID'],axis=1)
predicted_class=model.predict(test_X)
print(predicted_class)

my_submission = pd.DataFrame({'ID': test.ID, 'class': predicted_class})
my_submission.to_csv('submission.csv', index=False)