import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

warnings.filterwarnings('always')

# read csv file
train_data_df = pd.read_csv('../train_data.csv')
train_data_df['category'].value_counts(normalize=True).plot.bar()
plt.show()

# Create Feature and Label sets
X = train_data_df['text']
y = train_data_df['category']

# train test split (66% train - 33% test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
print(X_train.values.astype('U'))
print('Training Data :', X_train.shape)
print('Testing Data : ', X_test.shape)

# Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train.values.astype('U'))

# Hyperparameter Tuning for SVM
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# SVM Classifier
from sklearn.svm import SVC

svc = SVC()
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=15, n_jobs=-1, verbose=2)
grid_search.fit(X_train_cv, y_train)
print('Best Parameters : ', grid_search.best_params_)
print('Best Score : ', grid_search.best_score_)
print('Best Estimator : ', grid_search.best_estimator_)
# save all to grid_search_report.txt file
with open('reports/grid_search_report.txt', 'w') as f:
    f.write(f'Best Parameters : {grid_search.best_params_}\n')
    f.write(f'Best Score : {grid_search.best_score_}\n')
    f.write(f'Best Estimator : {grid_search.best_estimator_}\n')
    f.close()

# Fit the best estimator to the training data
svc = SVC(C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'],
            kernel=grid_search.best_params_['kernel'])
svc.fit(X_train_cv, y_train)

# save model to disk
import pickle
pickle.dump(svc, open('svm_classifier.pkl', 'wb'))
pickle.dump(cv, open('count_vectorizer.pkl', 'wb'))


