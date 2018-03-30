import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import math
import time


def findAccuracy(classes, normalized, level, feature, featuresSelected):
    knn = KNeighborsClassifier(n_neighbors=1)
    labels = classes
    correctPredicts = 0
    currentFeatures = featuresSelected[:]
    currentFeatures.append(feature)
    for row in range(0, len(labels) - 1):
        labels = classes
        if len(featuresSelected) == 0:
            train = np.array(normalized[:, currentFeatures[:]]).reshape(-1, 1)
            test1 = np.array(normalized[:, currentFeatures[:]]).reshape(-1, 1)
            test = np.array(test1[row]).reshape(-1, 1)
        else:
            train = np.array(normalized[:, currentFeatures[:]])
            test1 = np.array(normalized[row, currentFeatures[:]])
            test = []
            test.append(test1)
        testLabel = labels[row]
        labels = np.delete(labels, (row), axis=0)
        train = np.delete(train, (row), axis=0)
        knn.fit(train, labels)

        predictedClass = knn.predict(test)[0]
        if predictedClass == testLabel:
            correctPredicts = correctPredicts + 1

    accuracy = (float(correctPredicts) / float(len(classes))) * 100
    #print "accuracy", accuracy
    return accuracy

def betterAccuracyFinder(classes, normalized, bestAccSoFar, feature, featuresSelected):
    knn = KNeighborsClassifier(n_neighbors=3)
    bestIncorrectCount = 100 - bestAccSoFar
    labels = classes
    correctPredicts = 0
    currentFeatures = featuresSelected[:]
    incorrectPredicts = 0
    currentFeatures.append(feature)
    for row in range(0, len(labels) - 1):
        labels = classes
        if len(featuresSelected) == 0:
            train = np.array(normalized[:, currentFeatures[:]]).reshape(-1, 1)
            test1 = np.array(normalized[:, currentFeatures[:]]).reshape(-1, 1)
            test = np.array(test1[row]).reshape(-1, 1)
        else:
            train = np.array(normalized[:, currentFeatures[:]])
            test1 = np.array(normalized[row, currentFeatures[:]])
            test = []
            test.append(test1)
        testLabel = labels[row]
        labels = np.delete(labels, (row), axis=0)
        train = np.delete(train, (row), axis=0)
        knn.fit(train, labels)

        predictedClass = knn.predict(test)[0]
        if predictedClass == testLabel:
            correctPredicts = correctPredicts + 1
        else:
            incorrectPredicts = incorrectPredicts + 1
        if incorrectPredicts > bestIncorrectCount:
            return -1

    accuracy = (float(correctPredicts) / float(len(classes))) * 100
    print "accuracy", accuracy
    return accuracy


def forwardSelection(classes, normalized):
    start_time = time.time()
    currentFeatures = []
    rows, columns = normalized.shape
    finalBest = 0
    featureAtThisLevel = []
    labels = classes[:]
    for i in range(0, columns):
        bestAccSoFar = 0
        maxForLevel = 0
        for j in range(0, columns):
            if j in featureAtThisLevel:
                continue
            accuracy = findAccuracy(labels, normalized, i, j, featureAtThisLevel)
            levelFeatures = [x + 1 for x in featureAtThisLevel]
            levelFeatures.append(j+1)
            print "Using feature(s) ", levelFeatures, " accuracy is ", accuracy
            if accuracy > bestAccSoFar:
                bestAccSoFar = accuracy
                maxForLevel = j

        featureAtThisLevel.append(maxForLevel)
        if (bestAccSoFar > finalBest):
            finalBest = bestAccSoFar
            currentFeatures = featureAtThisLevel[:]
        else:
            print "Warning! Accuracy has decreased. Continuing search in case of local maxima."
        print_features = [x + 1 for x in featureAtThisLevel]
        print "Feature set ", print_features, " was best. Accuracy is ", bestAccSoFar
    finalFeatures = [x + 1 for x in currentFeatures]
    print "Finished Search!! Best feature subset was ", finalFeatures, ", accuracy is ", finalBest
    end_time = time.time()
    print "Time Taken: ", (end_time - start_time)


def backwardElimination(classes, normalized):
    start_time = time.time()
    featureAtThisLevel = []
    rows, columns = normalized.shape
    finalBest = 0
    for x in range(0,len(normalized[0,:])):
        featureAtThisLevel.append(x)
    currentFeatures = featureAtThisLevel[:]
    labels = classes[:]
    for i in range(0, columns):
        bestAccSoFar = 0
        maxForLevel = 0
        for j in range(0, len(currentFeatures)-1):
            delFeature = currentFeatures[j]
            del currentFeatures[j]
            accuracy = findAccuracy(labels, normalized, i, j, currentFeatures)
            levelFeatures = [x + 1 for x in currentFeatures]
            print "Using feature(s) ", levelFeatures, " accuracy is ", accuracy
            if accuracy > bestAccSoFar:
                bestAccSoFar = accuracy
                maxForLevel = j
            currentFeatures.insert(j, delFeature)
        del currentFeatures[maxForLevel]
        if bestAccSoFar > finalBest:
            finalBest = bestAccSoFar
            featureAtThisLevel = currentFeatures[:]
        else:
            print "Warning! Accuracy has decreased. Continuing search in case of local maxima."
        print_features = [x + 1 for x in currentFeatures]
        print "Feature set ", print_features, " was best. Accuracy is ", bestAccSoFar
    finalFeatures = [x + 1 for x in featureAtThisLevel]
    print "Finished Search!! Best feature subset was ", finalFeatures, ", accuracy is ", finalBest
    end_time = time.time()
    print "Time taken: ", (end_time - start_time)


def specialSearch(classes, normalized):
    start_time = time.time()
    currentFeatures = []
    rows, columns = normalized.shape
    finalBest = 0
    featureAtThisLevel = []
    labels = classes[:]
    for i in range(0, columns):
        bestAccSoFar = 0
        maxForLevel = 0
        for j in range(0, columns):
            if j in featureAtThisLevel:
                continue
            accuracy = betterAccuracyFinder(labels, normalized, finalBest, j, featureAtThisLevel)
            levelFeatures = [x + 1 for x in featureAtThisLevel]
            print "Using feature(s) ", levelFeatures, " accuracy is ", accuracy
            if accuracy > bestAccSoFar:
                bestAccSoFar = accuracy
                maxForLevel = j
            if accuracy == -1:
                continue
        featureAtThisLevel.append(maxForLevel)
        if (bestAccSoFar > finalBest):
            finalBest = bestAccSoFar
            currentFeatures = featureAtThisLevel[:]
        else:
            print "Warning! Accuracy has decreased. Continuing search in case of local maxima."
        print_features = [x + 1 for x in featureAtThisLevel]
        print "Feature set ", print_features, " was best. Accuracy is ", bestAccSoFar
    finalFeatures = [x + 1 for x in currentFeatures]
    print "Finished Search!! Best feature subset was ", finalFeatures, ", accuracy is ", finalBest
    end_time = time.time()
    print "Time Taken: ", (end_time - start_time)


def searchStart():
    print "Welcome to the Feature Selection Algorithm."
    print "Type in the name of the file to test: "
    fileName = raw_input()
    inputFile = open(fileName)
    rawData = inputFile.readlines()
    dataArray = [[]]
    classes = []
    for row in rawData:
        record = map(float, row.split())
        classes.append(record[0])
        del record[0]
        dataArray.append(record)

    del dataArray[0]
    print "Type the number of the algorithm you want to run."
    print "1)   Forward Selection"
    print "2)   Backward Elimination"
    print "3)   Special Search Algoithm"
    algorithmChoice = raw_input()
    instanceCount = len(dataArray)
    featureCount = len(dataArray[0])

    print "This dataset has ", featureCount, " features(not counting the class) with ", instanceCount, " instances."
    print "Please wait while I normalize the data."
    normalized = normalize_final(dataArray)
    npArray = np.array(normalized)
    if algorithmChoice == '1':
        forwardSelection(classes, npArray)

    if algorithmChoice == '2':
        backwardElimination(classes, npArray)

    if algorithmChoice == '3':
        specialSearch(classes, npArray)


def normalize_final(activeDataSet):
    dataSet = activeDataSet
    average = [0.00] * (len(dataSet[0]) - 1)
    stds = [0.00] * (len(dataSet[0]) - 1)
    #	get averages
    for i in dataSet:
        for j in range(1, (len(i))):
            average[j - 1] += i[j]
    for i in range(len(average)):
        average[i] = (average[i] / len(dataSet))
    #	get std's sqrt((sum(x-mean)^2)/n)
    for i in dataSet:
        for j in range(1, (len(i))):
            stds[j - 1] += pow((i[j] - average[j - 1]), 2)
    for i in range(len(stds)):
        stds[i] = math.sqrt(stds[i] / len(dataSet))
    #	calculate new values (x-mean)/std
    for i in range(len(dataSet)):
        for j in range(1, (len(dataSet[0]))):
            dataSet[i][j] = (dataSet[i][j] - average[j - 1]) / stds[j - 1]

    return dataSet


if __name__ == '__main__':
    searchStart()

# /Users/madhav/Documents/Acads/ArtificialIntelligence/P2/CS205_SMALLtestdata__58.txt
# /Users/madhav/Documents/Acads/ArtificialIntelligence/P2/CS205_BIGtestdata__19.txt
