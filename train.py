import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import joblib

# Store the variable we'll be predicting on.
target = "playtime_forever"

#use the whole training set and the real testing set
train = pd.read_csv("clean_train.csv")
test = pd.read_csv("clean_test.csv")

# games = pd.read_csv("clean_train.csv")
# train = games.sample(frac=0.8, random_state=1)
# test = games.loc[~games.index.isin(train.index)]


drop_cols = []
for i in train.columns:
    if (train[i].sum() <= 2):
        drop_cols.append(i)
train = train.drop(drop_cols, axis=1)
test = test.drop(drop_cols, axis=1)

# #correlation matrix
# corrmat = train.corr()
# fig = plt.figure(figsize = (12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True);
# plt.show()

print(test.shape)
print(train.shape)

# Get all the columns from the dataframe.
columns_test = test.columns.tolist()
columns_train = train.columns.tolist()
columns_test.remove('playtime_forever')
columns_train.remove('playtime_forever')

# from sklearn.linear_model import LinearRegression
# # Initialize the model class.
# model = LinearRegression()
# # Fit the model to the training data.
# model.fit(train[columns_train], train[target])
# # Generate our predictions for the test set.
# predictions = model.predict(test[columns_test])


# from sklearn.ensemble import RandomForestRegressor
# # Initialize the model with some parameters.
# model = RandomForestRegressor(n_estimators = 100)
# # Fit the model to the data.
# model.fit(train[columns_train], train[target])
# # Make predictions.
# predictions = model.predict(test[columns_test])


# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import VotingRegressor
# from sklearn import tree
# reg1 = GradientBoostingRegressor(n_estimators=100)
# reg2 = RandomForestRegressor(n_estimators=100)
# reg3 = tree.DecisionTreeRegressor()
# model = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('dt', reg3)])
# model.fit(train[columns_train], train[target])
# predictions = model.predict(test[columns_test])


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import VotingRegressor
# from sklearn import tree
# reg1 = RandomForestRegressor(n_estimators=100)
# reg2 = tree.DecisionTreeRegressor()
# model = VotingRegressor(estimators=[('rf', reg1), ('dt', reg2)])
# model.fit(train[columns_train], train[target])
# predictions = model.predict(test[columns_test])


from sklearn import tree
model = tree.DecisionTreeRegressor()
model.fit(train[columns_train], train[target])
predictions = model.predict(test[columns_test])



# from sklearn.ensemble import GradientBoostingRegressor
# model = GradientBoostingRegressor()
# model.fit(train[columns_train], train[target])
# predictions = model.predict(test[columns_test])



# from xgboost import XGBClassifier
# model = XGBClassifier()
# model.fit(train[columns_train], train[target])
# predictions = model.predict(test[columns_test])


# # load model
# model = joblib.load("rf_2_66.pkl")
# model.fit(train[columns_train], train[target])
# predictions = model.predict(test[columns_test])


# set negative to 0
predictions = np.where(predictions > 0, predictions, 0)


print(predictions)
np.savetxt('result.csv', predictions, delimiter = ' ')


# # Compute rmse between our test predictions and the actual values.
# print(mean_squared_error(test[target], predictions) ** 0.5)





