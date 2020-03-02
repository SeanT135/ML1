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

X_train, X_test, y_train, y_test = model_selection.train_test_split(csvDataX, csvDataY, test_size=0.25)

#print (datasets[0].size) 
#print (datasets[1].size)

clf = svm.SVC(kernel='rbf')

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

print('Accuracy : ', metrics.accuracy_score(y_test, y_predict))