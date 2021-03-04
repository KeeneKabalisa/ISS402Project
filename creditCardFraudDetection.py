#Download: Importing 
import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import LocalOutlierFactor

#Download: Dataset
filepth = os.path.dirname(os.path.abspath(sys.argv[0])) + "\creditcard.csv"

data = pd.read_csv(filepth)

fraud = data.loc[data['Class'] == 1]
valid = data.loc[data['Class'] == 0]

print("Number of valid cases: {}".format(len(valid)))
print("Number of fraudulent cases: {}".format(len(fraud)))

contam = len(fraud)/len(valid)

#Analysing the data
print("________ \n")
data.hist(figsize = (20,20))
plt.show()

corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = 0.8, square = True)
plt.show()


#Cleaning the data
data = data.sample(frac = 0.3, random_state = 1)

columns = data.columns.tolist()

columns = [c for c in columns if c not in ['Class']]

target = 'Class'

#All data except "Class" column
noClass = data[columns]

#Info in "Class" column
Y = data[target]


#Results
algorithm = LocalOutlierFactor(n_neighbors = 20, metric = "manhattan", contamination = contam)

predict = algorithm.fit_predict(noClass)

scores_pred = algorithm.negative_outlier_factor_

predict[predict == 1] = 0
predict[predict == -1] = 1

n_errors = (predict != Y).sum()

print("Local Outlier Factor: {} errors made".format(n_errors))
print("Accuracy percentage: {}".format(accuracy_score(Y, predict)*100))
print(classification_report(Y, predict))
