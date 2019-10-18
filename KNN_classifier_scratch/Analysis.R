
# Include tidyverse for ggplot
library(tidyverse)
#setwd("H:/Python/Project")
#Read the performance metrics from csv files and store them in dataframes
clKValues <- read.csv("python_k_values_accuracy.csv",header=FALSE, sep=",")
clAccuracy <- read.csv("python_classifier_accuracy.csv",header=FALSE, sep=",")
clPrecision <- read.csv("python_classifier_precision.csv",header=FALSE, sep=",")
clRecall <- read.csv("python_classifier_recall.csv",header=FALSE, sep=",")
clTime <- read.csv("python_classifier_time.csv",header=FALSE, sep=",")
libKValues <- read.csv("library_k_values_accuracy.csv",header=FALSE, sep=",")
libAccuracy <- read.csv("library_classifier_accuracy.csv",header=FALSE, sep=",")
libPrecision <- read.csv("library_classifier_precision.csv",header=FALSE, sep=",")
libRecall <- read.csv("library_classifier_recall.csv",header=FALSE, sep=",")
libTime <- read.csv("library_classifier_time.csv",header=FALSE, sep=",")

#k value from 1 to 20 - this is done to plot accuracy values with 20 parameters
k <- 1:20

#Plot accuracy values with KNN parameters - K = 1 to 20. this is to get teh best k value for the
classifier
ggplot(clKValues,aes(k,clKValues$V1))+
 geom_point()+
 geom_line() +
 xlab("Value of K for KNN") +
 ylab("Accuracy") +
 ggtitle("Classifier KNN parameters: Accuracy with different k values")

# Plot accuracy values for KNN parameetrs - for lbrary implementation
ggplot(libKValues,aes(k,libKValues$V1))+
 geom_point()+
 geom_line() +
 xlab("Value of K for KNN") +
 ylab("Accuracy") +
 ggtitle("Library KNN parameters: Accuracy with different k values")

# Perform a paired t-test for Accuracy - Library vs Programmed classifiers
cl_v_lib_Accuracy <- t.test(libAccuracy$V1,clAccuracy$V1,paired=TRUE,alternative="greater")
cl_v_lib_Accuracy

#Calculate the average of 100 accuracy values for library and programmed classifiers (Cross
validated)
meanLibAccuracy <- mean(libAccuracy$V1)
meanClAccuracy <- mean(clAccuracy$V1)

#Plot the accuracy values (Cross validated) - Library vs Programmed classifier
boxplot(libAccuracy$V1,clAccuracy$V1,names=c("library KNN","KNN classifier"),xlab
="Classifiers",ylab ="Accuracy",main="Accuracy: Library vs Python implementation")
# Perform a paired t-test for Precision - Library vs Programmed classifiers
cl_v_lib_Precision <- t.test(libPrecision$V1,clPrecision$V1,paired=TRUE,alternative="greater")
cl_v_lib_Precision

#Calculate the average of 100 precision values for library and programmed classifiers (Cross
validated)
meanLibPrecision <- mean(libPrecision$V1)
meanClPrecision <- mean(clPrecision$V1)

#Plot the precision values (Cross validated) - Library vs Programmed classifier
boxplot(libPrecision$V1,clPrecision$V1,names=c("library KNN","KNN classifier"),xlab
="Classifiers",ylab ="Precision",main="Precision: Library vs Python implementation")
# Perform a paired t-test for Recall - Library vs Programmed classifiers
cl_v_lib_Recall <- t.test(libRecall$V1,clRecall$V1,paired=TRUE,alternative="greater")
cl_v_lib_Recall

#Calculate the average of 100 recall values for library and programmed classifiers (Cross validated)
meanLibRecall <- mean(libRecall$V1)
meanClRecall <- mean(clRecall$V1)

#Plot the recall values (Cross validated) - Library vs Programmed classifier
boxplot(libRecall$V1,clRecall$V1,names=c("library KNN","KNN classifier"),xlab ="Classifiers",ylab
="Recall",main="Recall: Library vs Python implementation")

# Perform a paired t-test for Computational Time - Library vs Programmed classifiers
cl_v_lib_Time <- t.test(libTime$V1,clTime$V1,paired=TRUE,alternative="greater")
cl_v_lib_Time

#Calculate the average of 100 computatioanl times values for library and programmed classifiers
(Cross validated)
meanLibTime <- mean(libTime$V1)
meanClTime <- mean(clTime$V1)

#Plot the computational time values (Cross validated) - Library vs Programmed classifier
boxplot(libTime$V1,clTime$V1,names=c("library KNN","KNN classifier"),xlab ="Classifiers",ylab
="Computational Time",main="Computational Time: Library vs Python implementation")