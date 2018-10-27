import numpy as np
import svm as svm
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import os, sys
from scipy import stats

# File name to create data set based on training file or test file.
# target list for the y value: i.e. results (survivors) so this method is able to output
# two separate variables. A feature list (X) and a result list (y)
# Data amount refers to how much of the data we want to look at. The default is 1000 when a value is not given.
def loadData(fileName, target, data_Amount=10000000):
    occupancy = list()
    temperature = list()
    humidity = list()
    light = list()
    co2 = list()
    humidity_Ratio = list()

    skipFirst = True
    with open(fileName, "rt") as f:
        for line in f:
            if skipFirst:
                skipFirst = False
            elif data_Amount == 0:
                break
            else:
                try:
                    data_Amount -= 1
                    occupancy.append(float(line.split(',')[7]))
                    temperature.append(float(line.split(',')[2]))
                    humidity.append(float(line.split(',')[3]))
                    light.append(float(line.split(',')[4]))
                    co2.append(float(line.split(',')[5]))
                    humidity_Ratio.append(float(line.split(',')[6]))


                except ValueError:
                    print("Error parsing on line", line)


    # Test whether the Method works
    #print("The number of data stored in survived:\n", len(survived))
    #print("Expected: 1001, 1000 data plus 1 heading 'survived'.")

    # Test that all data have been stored into lists
    #print(len(Pclass), " ", Pclass)
    #print(len(parch), " ", parch)
    #print(len(fare), " ", fare)

    # merge into one feature set
    #le = preprocessing.LabelEncoder()
    #le.fit(is_match)
    #print(list(le.classes_))
    #fitted = le.transform(is_match)

    features = np.column_stack((temperature, humidity, light, co2, humidity_Ratio))
    #features.append(Pclass)
    #features.append(fitted.tolist())
    #features.append(parch)
    #features.append(fare)

    #print(len(features[1]), " ", features[1])
    target.extend(occupancy)
    return features

def SVM():
    train_y = list()
    train_X = loadData('datatraining.txt', train_y, 8000)
    print(len(train_y))
    print(len(train_X))

    test_y = list()
    test_X = loadData('datatest.txt', test_y)

    #print(train_X, '/n', test_X)
    sc = StandardScaler()
    sc.fit(train_X)

    train_X_std = sc.transform(train_X)
    test_X_std = sc.transform(test_X)

    model = svm.SVC(gamma=0.01)
    model.fit(train_X_std,train_y)

    y_pred = model.predict(test_X_std)

    print("Std accuracy: {0: .2f}%".format(accuracy_score(test_y, y_pred)*100))
    print("Std accuracy: {0: .4f}".format(accuracy_score(test_y, y_pred)))

    model2 = svm.SVC(gamma=0.01)
    model2.fit(train_X, train_y)

    y_pred_2 = model.predict(test_X)

    print("NonStd accuracy: {0: .2f}%".format(accuracy_score(test_y, y_pred_2) * 100))
    print("NonStd accuracy: {0: .4f}".format(accuracy_score(test_y, y_pred_2)))

    #print(train_y)
    #print(test_y)


if __name__ == '__main__':
    SVM()