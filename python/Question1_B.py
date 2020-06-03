from __future__ import division
import time
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def normalizeTrainDF(dataFrame):
    dfNormalized = dataFrame.copy()
    colList = list(dataFrame.columns)
    
    for col in range(len(colList)):
        colMean = dataFrame[colList[col]].mean()
        colStd = dataFrame[colList[col]].std()
        
        dfNormalized[colList[col]] = (dataFrame[colList[col]] - colMean)/colStd
    
    return dfNormalized

def normalizeTestDF(testDataFrame, trainDataFrame):

    dfNormalized = testDataFrame.copy()
    colList = list(testDataFrame.columns)

    for col in range(len(colList)):
        colMean = trainDataFrame[colList[col]].mean()
        colStd = trainDataFrame[colList[col]].std()
   
        dfNormalized[colList[col]] = (testDataFrame[colList[col]] - colMean)/colStd

    return dfNormalized
                     

def getAllDistanceDF(trainDF , testDF):
    indexCounter = 0;
    appenedData = []
    for row in testDF.itertuples(index=False, name='Pandas'):

        distanceSeries = calculateEculidDist(trainDF, row)
        testRowIndexList = [indexCounter]*np.shape(distanceSeries.values)[0]

        indexCounter += 1
    
        listToAddInFinalDF = { 'trainRowIndex' : np.array(distanceSeries.index) , 'distance': distanceSeries.values, 'testRowIndex' : testRowIndexList}
        dfEachTestRowDistance = pd.DataFrame(listToAddInFinalDF)
        appenedData.append(dfEachTestRowDistance)
    
    distanceDF = pd.concat(appenedData, axis=0)
    
    return distanceDF
    
def calculateEculidDist(trainDF , testRow):
  
    tmp = (((trainDF.sub( testRow, axis=1))**2).sum(axis=1))**0.5
    tmp.sort_values(axis=0, ascending=True, inplace=True)

    
    return tmp

startTime = time.clock()
trainDF = pd.read_csv("spam_train.csv")
testDF = pd.read_csv("spam_test.csv")

testDF.drop(testDF.columns[0] , axis=1, inplace=True)

trainLable = trainDF[['class']].copy()
testLable = testDF[['Label']].copy()

trainDF.drop('class' , axis=1, inplace=True)
testDF.drop('Label' , axis=1, inplace=True)

zScroreTime = time.clock()
trainDFNormalized = normalizeTrainDF(trainDF)  
testDFNormalized = normalizeTestDF(testDF,trainDF)
    
distanceDF = getAllDistanceDF(trainDFNormalized , testDFNormalized)
distanceTime = time.clock()

kValuesList= [1,5,11,21,41,61,81,101,201,401]

accuracyList = []
for kValue in range(len(kValuesList)) :
    loopStartTime = time.clock()
    indexCounter = 0
    predictedLabel = []
    for row in testDFNormalized.itertuples(index=False, name='Pandas'):
        distanceDFForRow = distanceDF[ distanceDF['testRowIndex'] == indexCounter]

        nnIndex = distanceDFForRow.loc[:(kValuesList[kValue] - 1) ,'trainRowIndex']
    
        tmp = trainLable.iloc[nnIndex]['class'].value_counts()
    
        predictedLabel.append(tmp.idxmax())
        indexCounter += 1
    
    tmpList = {'Label' : predictedLabel}
    predictedTestLabel = pd.DataFrame(tmpList)

    differenceLabel = testLable.sub(predictedTestLabel , axis=1)

    accurateClassCount = len(differenceLabel[ differenceLabel['Label'] ==0 ])

    accuracyPercent = accurateClassCount/testLable['Label'].count()*100

    print('The test accuracy for ' ,kValuesList[kValue] , 'is ' , (accurateClassCount/testLable['Label'].count())*100, '%' )
    
    accuracyList.append(accuracyPercent)

tempDict = {'KValue' : kValuesList, 'Accuray %' : accuracyList}
accuracyDF = pd.DataFrame(tempDict)
    
