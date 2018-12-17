import pandas as pd
import numpy as np
#from sklearn.linear_model import Perceptron
import svm as svm
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    desired_width = 320
    pd.set_option('display.width', desired_width)
    roomOcc = pd.read_csv("datatraining.txt")
    roomOcc = roomOcc.drop(["date"], axis=1)
    #print(roomOcc.head())
    #print(roomOcc.shape)
    d = roomOcc.describe()
    print(d)

    #corr = roomOcc.corr()
    #plt.figure(figsize=(10, 10))

    #sns.heatmap(corr, vmax=.8, linewidths=0.01,
    #            square=True, annot=True, cmap='Purples', linecolor="white")
    #plt.title('Correlation between features')

    roomOccTest = pd.read_csv("datatest.txt")
    roomOccTest = roomOccTest.drop(["date"], axis=1)
    #print(roomOccTest.head())

    y_train = roomOcc.pop('Occupancy').values
    y_test = roomOccTest.pop('Occupancy').values
    print(len(y_train))
    print(len(y_test))

    svmModel = svm.SVC(C=0.5)
    #Set the number of training data
    svmModel.fit(roomOcc[:17000],y_train[:17000])
    predict = svmModel.predict(roomOccTest)

    print(accuracy_score(y_test, predict))
