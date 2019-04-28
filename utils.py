import multiprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def test_over_models(x_train, y_train, x_test, y_test):
    acc_scores = []
    print('Training Logistic Regression...')
    logreg = LogisticRegression(n_jobs=multiprocessing.cpu_count(), C=1., solver='lbfgs', multi_class='multinomial')
    logreg.fit(x_train, y_train)
    print('Predicting topics...')
    y_pred = logreg.predict(x_test)
    acc_scores.append(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print('Training Gaussian Naive Bayes...')
    gnb = GaussianNB()
    gnb.fit(x_train if isinstance(x_train, np.ndarray) else x_train.todense(), y_train)
    print('Predicting topics...')
    y_pred = gnb.predict(x_test if isinstance(x_test, np.ndarray) else x_test.todense())
    acc_scores.append(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print('Training Rocchio Classifier...')
    rocchio = NearestCentroid()
    rocchio.fit(x_train, y_train)
    print('Predicting topics...')
    y_pred = rocchio.predict(x_test)
    acc_scores.append(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print('Training Decision Tree...')
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    print('Predicting topics...')
    y_pred = tree.predict(x_test)
    acc_scores.append(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print(acc_scores)



