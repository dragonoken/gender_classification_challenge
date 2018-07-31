from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Classifiers
clf_logreg = GridSearchCV(LogisticRegression(), param_grid={'C':[1, 10, 100, 1000, 10000]})
clf_svc = GridSearchCV(SVC(), param_grid={'C':[1, 10, 100, 1000, 10000],
                                          'gamma':[.1, .01, .001, .0001]})
clf_nb = GaussianNB()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Train them on the given data
clf_logreg.fit(X, Y)
clf_svc.fit(X, Y)
clf_nb.fit(X, Y)

prediction_logreg = clf_logreg.predict(X)
prediction_svc = clf_svc.predict(X)
prediction_nb = clf_nb.predict(X)

# Compare their reusults and print the best one

accuracies = {'Logistic Regression' : accuracy_score(Y, prediction_logreg),
              'Support Vector Machine' : accuracy_score(Y, prediction_svc),
              'Naive Bayes' : accuracy_score(Y, prediction_nb)}

print()
print('--- Accuracies ---')
print('Logistic Regression :', accuracies['Logistic Regression'])
print('Support Vector Machine :', accuracies['Support Vector Machine'])
print('Naive Bayes :', accuracies['Naive Bayes'])
print()
print('Best Classifier :', max(accuracies.keys(), key=lambda key: accuracies[key]))
print()
new_data = [[190, 70, 43]]
print("Predictions on", new_data[0])
print('Logistic Regression :', clf_logreg.predict(new_data)[0])
print('Support Vector Machine :', clf_svc.predict(new_data)[0])
print('Naive Bayes :', clf_nb.predict(new_data)[0])
