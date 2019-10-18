# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 20:39:17 2019

@author: malikk1
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, metrics
import pandas as pd
from sklearn.model_selection import cross_val_score

"""
Import file in a dataframe
Add column names to the data frame
"""
#Import file
breast_cancer_df = pd.read_csv('breast-cancer-wisconsin.data')

#Add column names
breast_cancer_df.columns = ['code_num','clump_thickness','uniform_cell_size',
                    'uniform_cell_shape','marginal_adhesion',
                    'single_epi_cell_size','bare_nuclei','bland_chromation',
                    'normal_nucleoli','mitoses','class']
#print(breast_cancer_df.head())


"""
2. Missing values:
    Replace ? in bare_nuclei column with the mode
"""

#Check occurence of '?'
print(breast_cancer_df['bare_nuclei'].value_counts())

#Find mode of stalk_root(11th) column
modeBareNuclei = str(breast_cancer_df['bare_nuclei'].mode()[0])

#Replace ? by the mode
breast_cancer_df = breast_cancer_df.replace("?", modeBareNuclei)

#Check values of column after replacement
print(breast_cancer_df['bare_nuclei'].value_counts())


"""
Drop the code_num column
"""
breast_cancer_df.drop(['code_num'],1,inplace=True)
#print(breast_cancer_df.head())


"""
Separating class columns from the dataset as it is a response variable
"""
X = np.array(breast_cancer_df.drop(['class'],1))
y = np.array(breast_cancer_df['class'])


"""
Create training and test samples
"""
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

"""
Train the K Neareast Neighbour classifier
"""
def KNN_train(k,X_train,y_train,X_test,y_test):
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict(X_test)
    """
    Test the classifier
    """
    knn_accuracy = metrics.accuracy_score(y_test, y_pred)
    knn_precision = metrics.precision_score(y_test, y_pred,average="macro")
    knn_recall = metrics.recall_score(y_test, y_pred,average="macro")
    #knn_accuracy = knn_clf.score(X_test, y_test)
    return knn_accuracy, knn_precision, knn_recall


def write_to_csv(filename, data):
    myFile = open (filename, "w")
    writer = csv.writer(myFile)
    writer.writerows(map(lambda x: [x], data))
    myFile.close()



"""
Testing
"""
#Set the value of k
k = 3

print("K Nearest neighbour classfier library implementation")
print("KNN result for k = " + str(k))
startTime = time.process_time()
accuracy,precision,recall = KNN_train(k,X_train,y_train,X_test,y_test)
totalTime = time.process_time() - startTime
print('Accuracy:', accuracy)
print('Precision:',precision)
print('Recall:',recall)
print('Computational Time:', totalTime)

k = 5
 
print("K Nearest neighbour classfier implementation")
print("KNN result for k = " + str(k)) 
startTime = time.process_time()  
accuracy,precision,recall = KNN_train(k,X_train,y_train,X_test,y_test)
totalTime = time.process_time() - startTime
print('Accuracy:', accuracy)
print('Precision:',precision)
print('Recall:',recall)
print('Computational Time:', totalTime)


"""
KNN parameters: Test accuracy with k values from 1 to 20
"""
kValues = list(range(1,21))

accuracyMatrix = []
for k in kValues:
    accuracy,precision,recall = KNN_train(k,X_train,y_train,X_test,y_test)
    accuracyMatrix.append(accuracy)
    
print("KNN results for k = 1 to 20")
print(accuracyMatrix)


"""
Cross validation:
    10-fold cross-validation for k nearest neighbours = 5
    k = 5
"""
k = 5
knnAccuracy = []
knnPrecision = []
knnRecall = []
knnComputationalTime = []
#knn_clf = neighbors.KNeighborsClassifier(n_neighbors=k)
#knnAccuracy.append(cross_val_score(knn_clf, X, y, cv=10, scoring='accuracy'))
#knnPrecision.append(cross_val_score(knn_clf, X, y, cv=10, scoring='precision_weighted'))
#knnRecall.append(cross_val_score(knn_clf, X, y, cv=10, scoring='recall_weighted'))
#
#print("Accuracy:", knnAccuracy)
#print("Precision:", knnPrecision)
#print("Recall:", knnRecall)

for i in range(100):
    X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
    startTime = time.process_time()
    accuracy,precision,recall = KNN_train(k,X_train,y_train,X_test,y_test)
    totalTime = time.process_time() - startTime
    knnAccuracy.append(accuracy)
    knnPrecision.append(precision)
    knnRecall.append(recall)
    knnComputationalTime.append(totalTime)

print("Accuracy:", knnAccuracy)
print("Precision:", knnPrecision)
print("Recall:", knnRecall)
print("Computational Time:",knnComputationalTime)


"""
Write metrics to csv file
"""
write_to_csv('library_k_values_accuracy.csv', accuracyMatrix)
write_to_csv('library_classifier_accuracy.csv', knnAccuracy)
write_to_csv('library_classifier_precision.csv', knnPrecision)
write_to_csv('library_classifier_recall.csv', knnRecall)
write_to_csv('library_classifier_time.csv', knnComputationalTime)
