#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn import preprocessing
from datetime import datetime

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from PandoraBDT import *

import os

# Utility function to move the midpoint of a colormap to be around
# the values of interest. - This is from rbf_gridsearch.py

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
		
if __name__=="__main__":

    trainingFile    = '../TestRegionTraining_nu_nue_nc_low_sampled_10k.txt'
    trainTestSplit    = 0.5

    # Load the data
    OverwriteStdout('Loading training set data from file ' + trainingFile + '\n')
    trainSet, nFeatures, nExamples = LoadData(trainingFile, ',')

    # Standardize the data and hold onto the means and stddevs for later
    OverwriteStdout(('Preprocessing ' + str(nExamples) + ' training examples of ' + 
                     str(nFeatures) + ' features'))
    
    X_org, Y_org     = SplitTrainingSet(trainSet, nFeatures)
    X_org, mu, sigma = StandardizeFeatures(X_org)
	
    # Train the BDT
    X, Y = Randomize(X_org, Y_org)
    X_train, Y_train, X_test, Y_test = Sample(X, Y, trainTestSplit)
	
    trees_range = np.arange(100, 550, 50) # every 50 trees
    depth_range = np.arange(1,11) # every 1 depth
    
    # to-do find cross validation for BDTs
    scores  = [[0 for x in range(10)] for y in range(9)]
    times  = [[0 for x in range(10)] for y in range(9)]
    sizes  = [[0 for x in range(10)] for y in range(9)]
	
    for i in range(0,9):
        for j in range(0,10):
            OverwriteStdout('Training AdaBoostClassifer...')
            bdtModel, trainingTime = TrainAdaBoostClassifer(X_train, Y_train, n_estimatorsValue=trees_range[i], max_depthValue=depth_range[j])
            xmlFileName = 'GridSearchRegion_NTrees_' + str(trees_range[i]) + '_TreeDepth_' + str(depth_range[j]) + '_low.xml'
            WriteXmlFile(xmlFileName, bdtModel)
            scores[i][j] = ValidateModel(bdtModel, X_test, Y_test)
            times[i][j] = trainingTime
            sizes[i][j] = os.path.getsize(xmlFileName)
            print("Ntrees %0.2f + depth %0.2f = score %0.2f, time %0.2f and size %0.2f" % (trees_range[i], depth_range[j], scores[i][j], times[i][j],sizes[i][j] ))
	
	
    #plt.figure(figsize=(8, 6))
    #plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    #plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
    #           norm=MidpointNormalize(vmin=0.2, midpoint=0.88))
	
    #plt.imshow(scores)
	
    #plt.xlabel('NumberOfTrees')
    #plt.ylabel('Depth')
    #plt.colorbar()
    #plt.title('Score')	
    #plt.xticks(np.arange(100, 550, 50), trees_range, rotation=45)
    #plt.xticks(np.arange(100, 550, 50),trees_range)
    #plt.yticks(depth_range)
    #plt.show()
    #plt.savefig('GridSearchRegion_Score.png')			
