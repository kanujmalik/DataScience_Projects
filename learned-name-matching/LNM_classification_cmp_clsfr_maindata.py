# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
test data
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

df_cl_smalltest = pd.read_csv('C:\\SITA\\small_label.csv')  #Loading training data
df_cl_test = df_cl_smalltest[['name1', 'name2', 'label']]


X_test = feature_engineering(df_cl_test)    #training set data
y_test = df_cl_test.label.values    #training set labels


result_cols = ["Classifier", "Accuracy"]
result_frame = pd.DataFrame(columns=result_cols)

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    LogisticRegression()]

for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test,predicted)
    print (name+' accuracy = '+str(acc*100)+'%')
    acc_field = pd.DataFrame([[name, acc*100]], columns=result_cols)
    result_frame = result_frame.append(acc_field)

    
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=result_frame, color="r")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()