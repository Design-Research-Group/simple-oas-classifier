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
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=7, n_jobs=-1, verbose=2)
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


# Predict the labels of the test set
y_pred = svc.predict(cv.transform(X_test.values.astype('U')))
print('Accuracy Score : ', accuracy_score(y_test, y_pred))
print('Classification Report : \n', classification_report(y_test, y_pred))

with open('reports/classification_report.txt', 'w') as f:
    f.write(f'Accuracy Score : {accuracy_score(y_test, y_pred)}\n')
    f.write(f'Classification Report : \n{classification_report(y_test, y_pred)}\n')
    f.close()

# plot classification report
plt.figure(figsize=(10, 10))
sns.heatmap(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).iloc[:-1, :].T, annot=True,
            cmap='Blues')
plt.show()

# plot accuracy score
plt.figure(figsize=(10, 10))
sns.barplot(x=['Accuracy Score'], y=[accuracy_score(y_test, y_pred)], palette='Blues')
plt.show()



# plot roc curve
from sklearn.metrics import roc_curve, roc_auc_score

y_pred_prob = svc.predict_proba(cv.transform(X_test.values.astype('U')))[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label='real')
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
plt.savefig('reports/roc_curve.png')

# plot precision recall curve
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob, pos_label='real')
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
plt.savefig('reports/precision_recall_curve.png')


# cross validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(svc, X_train_cv, y_train, cv=7, scoring='accuracy')
print('Cross Validation Accuracy Scores : ', scores)
print('Cross Validation Mean Score : ', scores.mean())

# save all to cross_validation_report.txt file
with open('reports/cross_validation_report.txt', 'w') as f:
    f.write(f'Cross Validation Accuracy Scores : {scores}\n')
    f.write(f'Cross Validation Mean Score : {scores.mean()}\n')
    f.close()

# plot learning curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(svc, X_train_cv, y_train, cv=7, scoring='accuracy',
                                                        n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Accuracy')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Validation Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('reports/learning_curve.png')



