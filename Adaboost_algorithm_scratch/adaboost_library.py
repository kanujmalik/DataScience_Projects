# https://www.datacamp.com/community/tutorials/adaboost-classifier-python

# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load data
breast_cancer_data = datasets.load_breast_cancer()
X = breast_cancer_data.data
y = breast_cancer_data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# AdaBoost classifier
adaboost = AdaBoostClassifier(n_estimators=200,
                         learning_rate=1)
# Train  Classifer
model = adaboost.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


