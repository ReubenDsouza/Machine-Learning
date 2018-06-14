import pandas as pd

data=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#data=data.fillna(data.mode())
data['Embarked']=data['Embarked'].fillna('S')
print(type(data))
print(data.isnull().any())
cols_with_missing = [col for col in data.columns 
                                 if data[col].isnull().any()]                                  
candidate_train_predictors = data.drop(['PassengerId', 'Survived'] + cols_with_missing, axis=1)
candidate_test_predictors = test.drop(['PassengerId'] + cols_with_missing, axis=1)


low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].dtype in ['int64', 'float64'] and
                                candidate_test_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
X = candidate_train_predictors[my_cols]
test_X = candidate_test_predictors[my_cols]

test_X=test_X.fillna(method='ffill')



Y=data.Survived

print(X.isnull().any())
print(test_X.isnull().any())


one_hot_encoded_training_predictors = pd.get_dummies(X)
one_hot_encoded_testing_predictors=pd.get_dummies(test_X)


from sklearn.preprocessing import Imputer

my_imputer = Imputer()

#imputed_train_X = my_imputer.fit_transform(X)


from sklearn.model_selection import train_test_split
#train_X, val_X, train_Y, val_Y = train_test_split(one_hot_encoded_training_predictors, Y,random_state = 0)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#model = SVC(C=1,kernel='rbf')
#model=ExtraTreesClassifier(min_samples_split=10)
#model=KNeighborsClassifier(n_neighbors=1)
#model=RandomForestClassifier(min_samples_split=5)
model=XGBClassifier()
#model.fit(imputed_train_X,Y)
model.fit(one_hot_encoded_training_predictors,Y)


#test_X = test[numeric_predictors.columns]
#test_Y=test.Survived

#imputed_test_X = my_imputer.transform(test_X)
# Use the model to make predictions
#predicted_survivors = model.predict(imputed_test_X)
predicted_survivors=model.predict(one_hot_encoded_testing_predictors)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_survivors)

#from sklearn.metrics import accuracy_score
#print(accuracy_score(val_Y,predicted_survivors))

my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predicted_survivors})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)










