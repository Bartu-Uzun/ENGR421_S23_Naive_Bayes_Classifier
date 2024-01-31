import numpy as np
import pandas as pd



X = np.genfromtxt("hw01_data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# first 50000 data points should be included to train
# remaining 43925 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    print(X.shape)
    X_train = X[: 50000];
    y_train = y[: 50000];
    X_test = X[50000 :];
    y_test = y[50000 :];
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below

    K = np.max(y_train); #K is the number of classes

    #if dpoint belongs to the class c+1, adding 1, then dividing by num of total dpoints
    
    # np.sum(<Condition>) --> +1 (True) for each element that satisfies the condition
    # np.mean(<Condition>) --> +1 (True) for each element that satisfies the condition, then divide by total # of elts
    # brackets on both ends to create an array
    class_priors = [np.mean(y_train == (c + 1)) for c in range(K)] 
    
    
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)


print(class_priors)



# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below

    D = X_train[0].size #get the number of features
    K = np.max(y_train); #K is the number of classes

    #print(X_train[y_train == 1]) # prints each datapoint with corresponding y = 1
    #print(X_train[y_train == 1, 0]) # prints the first letter of each datapoint w corresponding y = 1

    
    pAcd = np.array([[np.mean(X_train[y_train == (c + 1), k] == 'A') for k in range(D)] for c in range(K)]) 
    pCcd = np.array([[np.mean(X_train[y_train == (c + 1), k] == 'C') for k in range(D)] for c in range(K)]) 
    pGcd = np.array([[np.mean(X_train[y_train == (c + 1), k] == 'G') for k in range(D)] for c in range(K)]) 
    pTcd = np.array([[np.mean(X_train[y_train == (c + 1), k] == 'T') for k in range(D)] for c in range(K)]) 
    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)


print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below

    #N = np.size(X, 0) # number of datapoints, another way of calculating
    N = X.shape[0] # number of datapoints
    
    K = np.size(class_priors) # number of classes

    
    score_values = np.stack([[np.sum(np.log(pAcd[c, X[i] == 'A'])) + 
                              np.sum(np.log(pCcd[c, X[i] == 'C'])) + 
                              np.sum(np.log(pGcd[c, X[i] == 'G'])) + 
                              np.sum(np.log(pTcd[c, X[i] == 'T'])) + 
                              np.log(class_priors[c]) for c in range(K)] for i in range(N)])
    
  


    

    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below

    K = np.max(y_truth)

    # create an array of yi hats according to score values calculated
    y_predicted = np.array(np.repeat(1, len(y_truth)))
    y_predicted[scores[:, 0] < scores[:, 1]] = 2
    

    confusion_matrix = pd.crosstab(y_predicted, y_truth).to_numpy()
    
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
