import pandas as pd


train =pd.read_csv('train.csv')



from sklearn.ensemble import RandomForestRegressor



train_y = train.SalePrice
predictor_cols =train.drop(['SalePrice'], axis=1) 
print(predictor_cols)

numeric_predictors = predictor_cols.select_dtypes(exclude=['object'])

print(numeric_predictors)



from sklearn.preprocessing import Imputer

my_imputer = Imputer()





# Create training predictors data
train_X = train[numeric_predictors.columns]


imputed_train_X = my_imputer.fit_transform(train_X)






my_model = RandomForestRegressor()
my_model.fit(imputed_train_X, train_y)


test=pd.read_csv('test.csv')

test_X = test[numeric_predictors.columns]


imputed_test_X = my_imputer.transform(test_X)
# Use the model to make predictions
predicted_prices = my_model.predict(imputed_test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)