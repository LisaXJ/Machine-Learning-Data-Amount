import pandas as pd
import numpy as np
#from sklearn.linear_model import Perceptron
import svm as svm
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import seaborn as sns

if __name__ == '__main__':
    titanic = pd.read_csv("train.csv")
    titanic = titanic.drop(["Name", "PassengerId", "Fare"], axis=1)

    null_columns = titanic.columns[titanic.isnull().any()]
    #Sum of NULL values in training data
    #print(titanic.isnull().sum())
    print("\n")

    titanicTest = pd.read_csv("test.csv")
    titanicTest = titanicTest.drop(["Name", "PassengerId", "Fare"], axis=1)

    labelEnc = LabelEncoder()

    convert = ['Sex']
    for col in convert:
        titanic[col] = labelEnc.fit_transform(titanic[col])
        titanicTest[col] = labelEnc.fit_transform(titanicTest[col])

    null_columns = titanicTest.columns[titanic.isnull().any()]
    #Sum of NULL values in test data
    #print(titanicTest.isnull().sum())

    y_train = titanic.pop('Survived').values
    y_test = titanicTest.pop('Survived').values

    # 1 for male
    # 0 for female
    # 1 for survived
    # 0 for died
    # Hence the strong negative correlation. Males (1) where more likely to die (0)
    #corr = titanic.corr()
    #plt.figure(figsize=(10, 10))

    #sns.heatmap(corr, vmax=.8, linewidths=0.01,
                #square=True, annot=True, cmap='Blues', linecolor="white")
    #plt.title('Correlation between features: Titanic')

    #fig = plt.figure(figsize=(18, 6))
    #titanic.Sex.value_counts().plot(kind="bar", alpha=0.5)

    #titanic.drop(titanic.index[[100,999]])
    #y_train = np.delete(y_train, slice(100, 999), axis=0)

    #desc = titanic.describe()
    #print(desc)
    #print(titanicTest.head())


    svmModel = svm.SVC(C=0.5, gamma=0.75, kernel='rbf')
    svmModel.fit(titanic[:1000],y_train[:1000])
    predict = svmModel.predict(titanicTest)

    print(accuracy_score(y_test, predict))
