# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from collections import Counter
import random
import time
import csv


"""
Function to Calculate Eucledian distance between each pair of observations
Create a list. This will contain the distance list and the class for each data point
"""
def euc_distance(trainData, i):
    distance = []
    for classData in trainData:
        for item in trainData[classData]:
            euclideanDistance = math.sqrt((item[0]-i[0])**2 + (item[1]-i[1])**2)
            distance.append([euclideanDistance,classData])
    return distance

"""
Get the majority votes from the k nearest neighbours (k value set above)
"""
def k_neighbour(distance,k):
    #Sort the distance
    #Take the first k elements, taking just the class (index of 1)
    neighbour = [i[1] for i in sorted(distance)[:k]]
    
    #Get the most common vote using Collections
    vote = Counter(neighbour).most_common(1)[0][0]
    return vote


"""
Function to write lists to csv
"""
def write_to_csv(filename, data):
    myFile = open (filename, "w")
    writer = csv.writer(myFile)
    writer.writerows(map(lambda x: [x], data))
    myFile.close()


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
Missing values:
    Replace ? in bare_nuclei column with the mode
"""

#Check occurence of '?'
#print(breast_cancer_df['bare_nuclei'].value_counts())

#Find mode of stalk_root(11th) column
modeBareNuclei = str(breast_cancer_df['bare_nuclei'].mode()[0])

#Replace ? by the mode
breast_cancer_df = breast_cancer_df.replace("?", modeBareNuclei)

#Check values of column after replacement
#print(breast_cancer_df['bare_nuclei'].value_counts())


"""
Drop the code_num column
"""
breast_cancer_df.drop(['code_num'],1,inplace=True)
#print(breast_cancer_df.head())


breast_cancer_df = breast_cancer_df.astype(float).values.tolist()


"""
Split the data into training and test set
a. Shuffle the data randomly
b. 80-20 split
c. create dictionaries for trainig and test set havein 2 keys: 2 and 4
   2 for benign and 4 for malignant (from the dataset)
d. Populate the dictionaries, same as the train test split
"""
def train_test_split(data):
    
    #Shuffle the data randomly
    random.shuffle(breast_cancer_df)
    

    train_dict = {2:[], 4:[]}
    test_dict = {2:[], 4:[]}
    X_train = breast_cancer_df[:-int(0.2*len(breast_cancer_df))]
    X_test = breast_cancer_df[-int(0.2*len(breast_cancer_df)):]
    
    #Add attributes in the training dictionary, and class in the test dictionary
    for i in X_train:
        train_dict[i[-1]].append(i[:-1])
    
    for i in X_test:
        test_dict[i[-1]].append(i[:-1])
        
    return train_dict, test_dict
    
    
"""
Train the K Nearest neighbour model 
"""
def KNN_train(k, train, test):
    #Initialize the variables for correct outomes and total values.
    correctResult = 0
    totalValues = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    #For each class in the test dictionary, call Eucledian function and
    #vote function for each data item. Increment the correct result for every
    #correct result. Increment the total values traversed.
    #Calculate count of tru positives, true negatives, false positives, false negatives
    for classData in test_dict:
        for item in test_dict[classData]:
            distance = euc_distance(train_dict, item)
            vote = k_neighbour(distance, k)
            if classData == vote:
                correctResult += 1
                if classData == 2:
                    tp += 1
                else:
                    tn +=1
            else:
                if classData == 2:
                    fp += 1
                else:
                    fn += 1
            totalValues += 1
    accuracy = correctResult/totalValues
    return accuracy, tp, tn, fp, fn


#Set the value of k
k = 3

print("K Nearest neighbour classfier implementation")
train_dict, test_dict = train_test_split(breast_cancer_df)
print("KNN result for k = " + str(k))
startTime = time.process_time()
accuracy, tp, tn, fp, fn = KNN_train(k,train_dict,test_dict)
totalTime = time.process_time() - startTime
print('Accuracy:', accuracy)
print('Precision:', tp/(tp+fp))
print('Recall:', tp/(tp+fn))
print('Computational Time:', totalTime)



k = 5
print("K Nearest neighbour classfier implementation")
train_dict, test_dict = train_test_split(breast_cancer_df)
print("KNN result for k = " + str(k))
startTime = time.process_time()
accuracy, tp, tn, fp, fn = KNN_train(k,train_dict,test_dict)
totalTime = time.process_time() - startTime
print('Accuracy:', accuracy)
print('Precision:', tp/(tp+fp))
print('Recall:', tp/(tp+fn))
print('Computational Time:', totalTime)



"""
KNN parameters: Test accuracy with k values from 1 to 20
"""
kValues = list(range(1,21))

accuracyMatrix = []
for k in kValues:
    accuracy, tp, tn, fp, fn = KNN_train(k,train_dict,test_dict)
    accuracyMatrix.append(accuracy)
    
print("KNN results for k = 1 to 20")
print(accuracyMatrix)


"""
Cross validation:
    Resampling and running the model 100 times to get accuracy results
    k = 5
"""
k = 5
knnAccuracy = []
knnPrecision = []
knnRecall = []
knnComputationalTime = []
for i in range(100):    
    train_dict, test_dict = train_test_split(breast_cancer_df)
    startTime = time.process_time()
    accuracy, tp, tn, fp, fn = KNN_train(k,train_dict,test_dict)
    totalTime = time.process_time() - startTime
    knnAccuracy.append(accuracy)
    knnPrecision.append(tp/(tp+fp))
    knnRecall.append(tp/(tp+fn))
    knnComputationalTime.append(totalTime)
    
print("Cross validation: K Nearest neighbour classfier implementation")
print("100 KNN results for k = " + str(k))
print("Accuracy:", knnAccuracy)
print("Precision:", knnPrecision)
print("Recall:", knnRecall)
print("Computational Time:",knnComputationalTime)



"""
Write metrics to csv file
"""
write_to_csv('python_k_values_accuracy.csv', accuracyMatrix)
write_to_csv('python_classifier_accuracy.csv', knnAccuracy)
write_to_csv('python_classifier_precision.csv', knnPrecision)
write_to_csv('python_classifier_recall.csv', knnRecall)
write_to_csv('python_classifier_time.csv', knnComputationalTime)

    
    





