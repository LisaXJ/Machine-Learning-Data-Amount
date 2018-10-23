def loadData():
    survived = list()
    Pclass = list()
    sex = list()
    parch = list()
    fare = list()
    with open('train.csv', "rt") as f:
        for line in f:
            survived.append(line.split(',')[1])
            Pclass.append(line.split(',')[2])
            sex.append(line.split(',')[4])
            parch.append(line.split(',')[5])
            fare.append(line.split(',')[6])

    # Test whether the Method works
    print("The number of data stored in survived:\n", len(survived))
    print("Expected: 1001, 1000 data plus 1 heading 'survived'.")

    # Test that all data have been stored into lists
    print(len(Pclass))
    print(len(sex))
    print(len(parch))
    print(len(fare))


if __name__ == '__main__':
    loadData()