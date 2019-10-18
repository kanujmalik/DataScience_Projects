"""
SCC401: Distributed Artificial Intelligence
Final Project
Student ID: 35049410
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import metrics

#Define adaboost class
#takes the nunmber of base learners M as parameter.
class AdaBoost:
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y):
        self.models = [] #store decision stumps
        self.coeff = [] #store alpha

        N, _ = X.shape  #number of samples
        Wt = np.ones(N) / N  #uniform distribution - 1/N

    #loop: 1 to T
        for t in range(self.M):
            tree_stump = DecisionTreeClassifier(max_depth=1) #create new decision stump instance - dec tree_stump with max depth 1
        #fit the tree_stump to the data and pass in the sample weigh Wt
            tree_stump.fit(X,Y,sample_weight=Wt)
            P = tree_stump.predict(X)

    #vectorize form
            err = Wt.dot(P != Y)

            #alpha
            alpha = 0.5*(np.log(1-err) - np.log(err))

    #calculate weight updates
            Wt = Wt*np.exp(-alpha*Y*P)

    #normalize to sum to 1
            Wt  = Wt / Wt.sum()

            self.models.append(tree_stump)
            self.coeff.append(alpha)


    # predict
    def predict(self, X):
        N, _ = X.shape
        Hx = np.zeros(N)
        for alpha, tree_stump in zip(self.coeff, self.models):
            Hx +=alpha*tree_stump.predict(X)
        return np.sign(Hx), Hx

    #scre - calculates accuracy + loss
    def score(self, X, Y):
        P, Hx = self.predict(X)
        L = np.exp(-Y*Hx).mean()
        return np.mean(P==Y), L


#import data
def import_data():
    breast_cancer_data = datasets.load_breast_cancer()
    X = breast_cancer_data.data
    Y = breast_cancer_data.target
    return X,Y

#Main function
if __name__ == '__main__':
    X, Y  = import_data()
    Y[Y == 0] = -1  # make the targets -1, +1
    trainData = int(0.8*len(X))
    Xtrain, Ytrain = X[:trainData], Y[:trainData]
    Xtest, Ytest = X[trainData:], Y[trainData:]
    
    T = 200
    train_errors =np.empty(T)
    test_losses =np.empty(T)
    test_errors = np.empty(T)
    

    # if tree_num %20 is 0, we print out tree_num
    for tree_num in range(T):
        if tree_num == 0:
            train_errors[tree_num] = None
            test_errors[tree_num] = None
            test_losses[tree_num] = None
            continue
        if tree_num % 20  == 0:
            print(tree_num)

        #create AdaBoost instance
        model  = AdaBoost(tree_num)

        #Fit the training data
        model.fit(Xtrain, Ytrain)
        accuracy, loss = model.score(Xtest, Ytest)
        train_accuracy, _ = model.score(Xtrain, Ytrain)

        #Errors = 1-accuracy
        train_errors[tree_num] = 1 - train_accuracy
        test_errors[tree_num] = 1 - accuracy
        test_losses[tree_num] = loss
        
        
        #Print final train nd test error
        if tree_num == T - 1:
            print("final train error:", 1 - train_accuracy)
            print("final test error:", 1 - accuracy)
    
    #Plot test errors v test losses
    plt.plot(test_errors, label = 'test_errors')
    plt.plot(test_losses, label = 'test losses')
    plt.legend()
    plt.show()
    
    #Plot train errors v test errors
    plt.plot(train_errors, label = 'train_errors')
    plt.plot(test_errors, label = 'test errors')
    plt.legend()
    plt.show()
    
