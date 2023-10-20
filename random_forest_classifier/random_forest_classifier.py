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

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 30, 50, 70, 100],
    'max_depth': [None, 3, 5, 7],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 3, 5]
}

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=7, n_jobs=-1, verbose=2)
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

# Final Model using grid search best parameters
rfc = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                             max_depth=grid_search.best_params_['max_depth'],
                             max_features=grid_search.best_params_['max_features'],
                             min_samples_leaf=grid_search.best_params_['min_samples_leaf'])
rfc.fit(X_train_cv, y_train)


import pickle
pickle.dump(rfc, open('random_forest_classifier.pkl', 'wb'))
pickle.dump(cv, open('count_vectorizer.pkl', 'wb'))

y_pred = rfc.predict(cv.transform(X_test.values.astype('U')))
print('Accuracy Score : ', accuracy_score(y_test, y_pred))
print('Classification Report : \n', classification_report(y_test, y_pred))
print('Confusion Matrix : \n', confusion_matrix(y_test, y_pred))

# save all to final_model_report.txt file
with open('reports/final_model_report.txt', 'w') as f:
    f.write(f'Accuracy Score : {accuracy_score(y_test, y_pred)}\n')
    f.write(f'Classification Report : \n{classification_report(y_test, y_pred)}\n')
    f.write(f'Confusion Matrix : \n{confusion_matrix(y_test, y_pred)}\n')
    f.close()

# plot confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
plt.savefig('reports/confusion_matrix.png')

# plot classification report
plt.figure(figsize=(10, 10))
sns.heatmap(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).iloc[:-1, :].T, annot=True,
            cmap='Blues')
plt.show()
plt.savefig('reports/classification_report.png')

# plot accuracy score
plt.figure(figsize=(10, 10))
sns.barplot(x=['Accuracy Score'], y=[accuracy_score(y_test, y_pred)], palette='Blues')
plt.show()
plt.savefig('reports/accuracy_score.png')




# cross validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rfc, X_train_cv, y_train, cv=7, scoring='accuracy', n_jobs=-1)
print('Cross Validation Scores : ', cv_scores)
print('Cross Validation Mean Score : ', cv_scores.mean())

# save all to cross_validation_report.txt file
with open('reports/cross_validation_report.txt', 'w') as f:
    f.write(f'Cross Validation Scores : {cv_scores}\n')
    f.write(f'Cross Validation Mean Score : {cv_scores.mean()}\n')
    f.close()



# plot learning curve
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(rfc, X_train_cv, y_train, cv=7, scoring='accuracy', n_jobs=-1,
                                                        train_sizes=np.linspace(0.01, 1.0, 50))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, label='Training Score', color='blue')
plt.plot(train_sizes, test_mean, label='Cross Validation Score', color='red')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='red', alpha=0.1)
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Accuracy Score')
plt.legend()
plt.show()
plt.savefig('reports/learning_curve.png',)

# Load the model
rfc_model = pickle.load(open('random_forest_classifier.pkl', 'rb'))
cv_model = pickle.load(open('count_vectorizer.pkl', 'rb'))

# Test the model
test_text = [
    'news for this city is not available, get news for another city, please. posts on wall of user. get all users']
test_text_cv = cv_model.transform(test_text)
test_pred = rfc_model.predict(test_text_cv)
print(test_pred)
