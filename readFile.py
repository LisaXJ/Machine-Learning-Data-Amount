import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import os, sys
from scipy import stats

# File name to create data set based on training file or test file.
# target list for the y value: i.e. results (survivors) so this method is able to output
# two separate variables. A feature list (X) and a result list (y)
# Data amount refers to how much of the data we want to look at. The default is 1000 when a value is not given.
def loadData(fileName, target, data_Amount=1000):
    survived = list()
    Pclass = list()
    sex = list()
    parch = list()
    fare = list()
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
                    survived.append(float(line.split(',')[1]))
                    Pclass.append(float(line.split(',')[2]))
                    sex.append(line.split(',')[5])
                    parch.append(float(line.split(',')[6]))
                    f = line.split(',')[7].rstrip()
                    fare.append(float(f))
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
    le = preprocessing.LabelEncoder()
    le.fit(sex)
    #print(list(le.classes_))
    fitted = le.transform(sex)

    features = np.column_stack((Pclass, fitted.tolist(), parch, fare))
    #features.append(Pclass)
    #features.append(fitted.tolist())
    #features.append(parch)
    #features.append(fare)

    #print(len(features[1]), " ", features[1])
    target.extend(survived)
    return features

def Per():
    train_y = list()
    train_X = loadData('train.csv', train_y, 300)
    print(len(train_y))
    print(len(train_X))

    test_y = list()
    test_X = loadData('test.csv', test_y)

    #print(train_X, '/n', test_X)
    sc = StandardScaler()
    sc.fit(train_X)

    train_X_std = sc.transform(train_X)
    test_X_std = sc.transform(test_X)

    ppn = Perceptron()
    ppn.fit(train_X_std,train_y)

    y_pred = ppn.predict(test_X_std)

    print("accuracy: {0: .2f}%".format(accuracy_score(test_y, y_pred)*100))

    #print(train_y)
    #print(test_y)


if __name__ == '__main__':
    Per()