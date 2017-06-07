#!/usr/bin/env python
# PandoraSVM.py

from sklearn import svm
from sklearn import preprocessing
from datetime import datetime

import numpy as np
import sys
import time
import pickle

def LoadData(trainingFileName, delimiter=','):
    # Use the first example to get the number of columns
    with open(trainingFileName) as file:
        ncols = len(file.readline().split(delimiter))
        
    # First column is a datestamp, so skip it
    trainingSet = np.genfromtxt(trainingFileName, delimiter=delimiter, usecols=range(1,ncols), 
                                dtype=None)
                                
    nExamples = trainingSet.size
    nFeatures = ncols - 2 # last column is the response
    
    return np.array(trainingSet), nFeatures, nExamples
    
#--------------------------------------------------------------------------------------------------

def SplitTrainingSet(trainingSet, nFeatures):
    X=[] # features sets
    Y=[] # responses

    for example in trainingSet:
        Y.append(int(example[nFeatures])) # type of Y should be bool or int
        features = []
        for i in range(0, nFeatures):
            features.append(float(example[i])) # features in this SVM must be Python float
            
        X.append(features)

    return np.array(X).astype(np.float64), np.array(Y).astype(np.int)
    
#--------------------------------------------------------------------------------------------------

def StandardizeFeatures(X):
    muValues    = np.mean(X, axis=0)
    sigmaValues = np.std(X, axis=0)
    return np.divide((X - muValues), sigmaValues), muValues, sigmaValues
    
#--------------------------------------------------------------------------------------------------

def Randomize(X, Y, setSameSeed=False):
    if setSameSeed:
        np.random.seed(0)

    order = np.random.permutation(Y.size)
    return X[order], Y[order]
    
#--------------------------------------------------------------------------------------------------

def Sample(X, Y, testFraction=0.1):
    trainSize = int((1.0 - testFraction) * Y.size)
    
    X_train = X[:trainSize]
    Y_train = Y[:trainSize]
    X_test  = X[trainSize:]
    Y_test  = Y[trainSize:]
    
    return X_train, Y_train, X_test, Y_test
    
#--------------------------------------------------------------------------------------------------

def TrainModel(X_train, Y_train, kernelString, kernelDegree=2, gammaValue=0.05, coef0Value=1.0, 
               cValue=1.0, tol=0.001, cache_size=1000):
    # Load the SVC object
    svmModel = svm.SVC(C=cValue, cache_size=cache_size, class_weight=None, coef0=coef0Value,
                       decision_function_shape=None, degree=kernelDegree, gamma=gammaValue, 
                       kernel=kernelString, max_iter=-1, probability=False, random_state=None, 
                       shrinking=True, tol=tol, verbose=False)
    
    # Train the model   
    startTime = time.time() 
    svmModel.fit(X_train, Y_train)
    
    endTime = time.time()
    nSupportVectors = svmModel.support_vectors_.shape[0]
    return svmModel, endTime - startTime, nSupportVectors
    
#--------------------------------------------------------------------------------------------------

def ValidateModel(svmModel, X_test, Y_test):               
    return svmModel.score(X_test, Y_test)

#--------------------------------------------------------------------------------------------------

def QuickTest(X_train, Y_train, X_test, Y_test, kernelString, kernelDegree=2, gammaValue=0.05, 
              coef0Value=1.0, cValue=1.0):
    # Train and validate the model
    svmModel, trainingTime, nSupportVectors = TrainModel(X_train, Y_train, kernelString, 
                                                         kernelDegree, gammaValue, coef0Value, 
                                                         cValue)
                                                         
    modelScore = ValidateModel(svmModel, X_test, Y_test)
    
    # Write validation output to screen
    stdoutString = '[' + kernelString
    if kernelString == 'poly':
        stdoutString += ',deg=' + str(kernelDegree)
        
    elif kernelString == 'rbf':
        stdoutString += ',gamma=' + str(gammaValue)
        
    if kernelString != 'rbf' and kernelString != 'linear':
        stdoutString += ',coef0=' + str(coef0Value)

    stdoutString += (',C=' + str(cValue) + '] : %.1f%% (%d seconds, %d SVs)\n' % 
                     (modelScore * 100, trainingTime, nSupportVectors))
                     
    OverwriteStdout(stdoutString)
    
#--------------------------------------------------------------------------------------------------

def OverwriteStdout(text):
    sys.stdout.write('\x1b[2K\r' + text)
    sys.stdout.flush()

#--------------------------------------------------------------------------------------------------

def OpenXmlTag(modelFile, tag, indentation):
    modelFile.write((' ' * indentation) + '<' + tag + '>\n')
    return indentation + 4
    
#--------------------------------------------------------------------------------------------------

def CloseXmlTag(modelFile, tag, indentation):
    indentation = max(indentation - 4, 0)
    modelFile.write((' ' * indentation) + '</' + tag + '>\n')
    return indentation

#--------------------------------------------------------------------------------------------------

def WriteXmlFeatureVector(modelFile, featureVector, tag, indentation):
    modelFile.write((' ' * indentation) + '<' + tag + '>')

    firstTime=True
    for feature in featureVector:
        if firstTime:
            modelFile.write(str(feature))
            firstTime=False
        else:
            modelFile.write(' ' + str(feature))
            
    modelFile.write('</' + tag + '>\n')
    
#--------------------------------------------------------------------------------------------------

def WriteXmlFeature(modelFile, feature, tag, indentation):
    modelFile.write((' ' * indentation) + '<' + tag + '>')
    modelFile.write(str(feature))     
    modelFile.write('</' + tag + '>\n')

#--------------------------------------------------------------------------------------------------

def WriteXmlFile(filePath, svmName, datetimeString, yAlpha, bias, kernel, mu, scale, sigma, supportVectors, standardize=True):
    with open(filePath, "a") as modelFile:
        standStr = str(standardize).lower()
        indentation = OpenXmlTag(modelFile,    'SupportVectorMachine', 0)
        WriteXmlFeature(modelFile, svmName,        'Name', indentation)
        WriteXmlFeature(modelFile, datetimeString, 'Timestamp', indentation)
        
        indentation = OpenXmlTag(modelFile,        'Machine', indentation)
        WriteXmlFeature(modelFile, kernel,             'KernelType', indentation)
        WriteXmlFeature(modelFile, bias,               'Bias', indentation)
        WriteXmlFeature(modelFile, scale,              'ScaleFactor', indentation)
        WriteXmlFeature(modelFile, standStr,           'Standardize', indentation)
        indentation = CloseXmlTag(modelFile,       'Machine', indentation)
        
        indentation = OpenXmlTag(modelFile,        'Features', indentation)
        WriteXmlFeatureVector(modelFile, mu,           'MuValues', indentation)
        WriteXmlFeatureVector(modelFile, sigma,        'SigmaValues', indentation)
        indentation = CloseXmlTag(modelFile,       'Features', indentation)
        
        for supVec, yAlphaValue in zip(supportVectors, yAlpha):
            indentation = OpenXmlTag(modelFile,    'SupportVector', indentation)
            WriteXmlFeature(modelFile, yAlphaValue,    'AlphaY', indentation)
            WriteXmlFeatureVector(modelFile, supVec,   'Values', indentation)
            indentation = CloseXmlTag(modelFile,   'SupportVector', indentation)
        
        CloseXmlTag(modelFile,                 'SupportVectorMachine', indentation)

#--------------------------------------------------------------------------------------------------

def GetKernelInt(kernelType, kernelDegree=2):   
    if kernelType == 'linear':
        return 1
        
    if kernelType == 'poly' and kernelDegree == 2:
        return 2
        
    if kernelType == 'poly' and kernelDegree == 3:
        return 3
        
    if kernelType == 'rbf':
        return 4

    raise ValueError('Unknown kernel type for Pandora kernel enum: ' + kernelType)
    
#--------------------------------------------------------------------------------------------------
    
def SerializeToPkl(fileName, svmModel, mu, sigma):
    with open(fileName, 'w') as f:
        pickle.dump(svmModel, f)
        pickle.dump(mu, f)
        pickle.dump(sigma, f)
    
#--------------------------------------------------------------------------------------------------
    
def LoadFromPkl(fileName, svmModel, mu, sigma):
    with open(fileName, 'r') as f:
        svmModel = pickle.load(f) 
        mu       = pickle.load(f)
        sigma    = pickle.load(f)
        
        return svmModel, mu, sigma
