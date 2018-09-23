#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
import logging
import time
import datetime

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
print("./log/" + st + ".log")

logger = logging.getLogger('myapp')
hdlr = logging.FileHandler("./log/" + st + ".log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.DEBUG)

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print(len(features_train[0]))
#########################################################
### your code goes here ###
clf = tree.DecisionTreeClassifier(min_samples_split=40)
logger.info("Training starttime:" + str(time.time()))
clf = clf.fit(features_train,labels_train)
logger.info("Training endtime" + str(time.time()))
score = clf.score(features_test,labels_test)
logger.info("Prediction done:" + str(time.time()) + " score " + str(score))

#########################################################


