import pandas as pd
import numpy as np


# Read in the data.
test = pd.read_csv("test.csv")
# Print the names of the columns in test.
# print(test.columns)
# print(test.shape)

# columns = ['id', 'playtime_forever', 'is_free', 'price', 'genres', 'categories',
#        'tags', 'purchase_date', 'release_date', 'total_positive_reviews',
#        'total_negative_reviews']


# set the dummies_genres
dummies_genres = test.loc[:,"genres"].str.get_dummies(sep=',')
test = pd.concat([test, dummies_genres], axis=1)
test = test.drop(["genres"], axis = 1 )

# set the dummies_categories
dummies_categories = test.loc[:,"categories"].str.get_dummies(sep=',')
test = pd.concat([test, dummies_categories], axis=1)
test = test.drop(["categories"], axis = 1 )

# set the dummies_tags
dummies_tags = test.loc[:,"tags"].str.get_dummies(sep=',')
test = pd.concat([test, dummies_tags], axis=1)
test = test.drop(["tags"], axis = 1 )

# transform object to datetime
datel = []
for x in test.columns.tolist():
    if 'date' in x:
        datel.append(x)
test[datel]=test[datel].apply(pd.to_datetime)

# fill in missing data
test['purchase_date'] = test.purchase_date.fillna(method='bfill')
test['release_date'] = test.release_date.fillna(method='bfill')

# function that extracts year month day
def extract_date(df, column):
    df[column + '_year'] = df[column].apply(lambda x: x.year)
    df[column + '_month'] = df[column].apply(lambda x: x.month)
    df[column + '_day'] = df[column].apply(lambda x: x.day)

extract_date(test, 'purchase_date')
extract_date(test, 'release_date')


# drop some columns
test = test.drop(["id"], axis = 1 )
test = test.drop(["is_free"], axis = 1 )
test = test.drop(["purchase_date"], axis = 1 )
test = test.drop(["release_date"], axis = 1 )



# Read in the data.
train = pd.read_csv("train.csv")

# set the dummies_genres
dummies_genres = train.loc[:,"genres"].str.get_dummies(sep=',')
train = pd.concat([train, dummies_genres], axis=1)
train = train.drop(["genres"], axis = 1 )

# set the dummies_categories
dummies_categories = train.loc[:,"categories"].str.get_dummies(sep=',')
train = pd.concat([train, dummies_categories], axis=1)
train = train.drop(["categories"], axis = 1 )

# set the dummies_tags
dummies_tags = train.loc[:,"tags"].str.get_dummies(sep=',')
train = pd.concat([train, dummies_tags], axis=1)
train = train.drop(["tags"], axis = 1 )

# transform object to datetime
dater = []
for x in train.columns.tolist():
    if 'date' in x:
        dater.append(x)
train[dater]=train[dater].apply(pd.to_datetime)

# fill in missing data
train['purchase_date'] = train.purchase_date.fillna(method='bfill')
train['release_date'] = train.release_date.fillna(method='bfill')

# function that extracts year month day
def extract_date(df, column):
    df[column + '_year'] = df[column].apply(lambda x: x.year)
    df[column + '_month'] = df[column].apply(lambda x: x.month)
    df[column + '_day'] = df[column].apply(lambda x: x.day)

extract_date(train, 'purchase_date')
extract_date(train, 'release_date')


train = train.drop(["id"], axis = 1 )
train = train.drop(["is_free"], axis = 1 )
train = train.drop(["purchase_date"], axis = 1 )
train = train.drop(["release_date"], axis = 1 )


# merge repeated columns
train = train.groupby(level=0, axis=1).first()
test = test.groupby(level=0, axis=1).first()

print(test.shape)
print(train.shape)

for j in test.columns:
    if (j not in train.columns):
        train[j] = np.nan

for i in train.columns:
    if (i not in test.columns):
        test[i] = np.nan

# adjust to same sequence
test = test[train.columns]

#set nan to 0
test = test.fillna(0)
train = train.fillna(0)

print(test.shape)
print(train.shape)

test.to_csv("clean_test.csv",index=False)
train.to_csv("clean_train.csv",index=False)