# -*- coding: utf-8 -*-

from matplotlib import *
from pandas import *
from sklearn import *

csvData = pandas.read_csv("messidor_features.csv", sep=',', header=0, index_col=False, skipinitialspace=True)
# Parameters: Filepath, Seperating Character, Header Row, Index Column (set to false, as no index column), Skip Initial Space (Ignores whitespace after seperationg character)

csvDataX = csvData
csvDataX.drop(index=19)
csvDataY = csvData['Messidor Class']


#print (csvData.size) 

jkCount = 0 # Jackknife Method counter for loop

# TP: True Positive
# FP: False Positive
# TN: True Negative
# FN: False Negative


# Accuracy Scores: Accuracy = (TP+TN)/(TP+FP+FN+TN)
# Compared to all values, how often are true values evaluated?
totAccScore = 0 # Total Accuracy score w/o Feature Scaling
totScaledAccScore = 0 # Total Accuracy score w/ Feature Scaling

# Precision Scores: Precision = (TP)/(TP+FP)
# Compared to all positives, how often are true positives evaluated
totPrecScore = 0 # Total Precision score w/o Feature Scaling
totScaledPrecScore = 0 # Total Precision score w/ Feature Scaling


# Recall Scores: Recall = (TP)/(TP+FN)
# Compared to all values that should be positive, how often are they evaluated as positive
totRecScore = 0 # Total Recall score w/o Feature Scaling
totScaledRecScore = 0 # Total Recall score w/ Feature Scaling


while jkCount < 10:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(csvDataX, csvDataY, test_size=0.25)

#print (datasets[0].size) 
#print (datasets[1].size)


    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    X_scaled_train = scaler.transform(X_train)
    X_scaled_test = scaler.transform(X_test)

    clf = neural_network.MLPClassifier(max_iter = 1000)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    
    clf.fit(X_scaled_train, y_train)
    y_scaled_predict = clf.predict(X_scaled_test)

    #accScore[jkCount] = metrics.accuracy_score(y_test, y_predict)
    totAccScore += metrics.accuracy_score(y_test, y_predict)
    totScaledAccScore += metrics.accuracy_score(y_test, y_scaled_predict)
    
    totPrecScore += metrics.precision_score(y_test, y_predict)
    totScaledPrecScore += metrics.precision_score(y_test, y_scaled_predict)
    
    totRecScore += metrics.recall_score(y_test, y_predict)
    totScaledRecScore += metrics.recall_score(y_test, y_scaled_predict)
    
    jkCount += 1

avgAcc = totAccScore / 10
avgScaledAcc = totScaledAccScore /10

avgPrec = totPrecScore / 10
avgScaledPrec = totScaledPrecScore /10

avgRec = totRecScore / 10
avgScaledRec = totScaledRecScore /10

print('Average Accuracy w/o Feature Scaling : ', avgAcc)
print('Average Accuracy w/ Feature Scaling : ', avgScaledAcc)

print('Average Precision w/o Feature Scaling : ', avgPrec)
print('Average Precision w/ Feature Scaling : ', avgScaledPrec)

print('Average Recall w/o Feature Scaling : ', avgRec)
print('Average Recall w/ Feature Scaling : ', avgScaledRec)