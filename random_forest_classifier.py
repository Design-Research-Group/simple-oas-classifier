import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# read csv file
train_data_df = pd.read_csv('train_data.csv')
train_data_df['category'].value_counts(normalize=True).plot.bar()
plt.show()

# Create Feature and Label sets

X = train_data_df['text']
print(X.array)


y = train_data_df['category']
# train test split (66% train - 33% test)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

print('Training Data :', X_train.shape)
print('Testing Data : ', X_test.shape)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
print(X_train_cv.shape)

# Training Logistic Regression model
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(max_iter=500)
# lr.fit(X_train_cv, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X, y)

# transform X_test using CV
#
# X_test_cv = cv.transform(X_test)

# generate predictions

# predictions = classifier.predict(X_test_cv)
#
# print(predictions)
#
# from sklearn import metrics
#
# df = pd.DataFrame(metrics.confusion_matrix(y_test, predictions), index=list(set(y)), columns=list(set(y)))
#
# print(df)
