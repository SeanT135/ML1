# -*- coding: utf-8 -*-

import matplotlib
import pandas
import scikit_learn

csvData = pandas.read_csv("messidor_features.csv", sep=',', header=0, index_col=False, skipinitialspace=True)
# Parameters: Filepath, Seperating Character, Header Row, Index Column (set to false, as no index column), Skip Initial Space (Ignores whitespace after seperationg character) 
