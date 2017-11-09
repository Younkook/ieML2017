import numpy as np
import pandas as pd
import tensorflow as tf

path = "C:\\Users\\KangYounKook\\Desktop\\ML_Study\\"
filename = "sectionInfo.txt"

def testss():
    print('testss')
    return
    
def getDatasetFromFile(filename):
    dataset = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            words = line.strip('\n').split(',')
    
            sectionId = words[0]
            sectionType = words[1]
            distance = words[3]
            speed = words[4]
    
            incount = int(words[5])
            incomings = []
            for i in range(6, 6+incount):
                incomings.append(words[i])
        #    print(incomings)
            outcount = int(words[6+incount])
            outgoings = []
            for i in range(7+incount, 7+incount+outcount):
                outgoings.append(words[i])
        #    print(outgoings)
    
            value = [sectionType, distance, speed, incomings, outgoings]
            dataset[sectionId] = value
            #printDataset(sectionId)
        print('getDataFromFile ',filename)
    return dataset

def printDataset(sectionId):
    print(sectionId)
    print(dataset[sectionId])

def updateDatasetFromFile(filename, dataset):
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip('\n').split(',')
            sectionId = words[0]
            move_count = int(words[6])
            work_count = int(words[7])
            ratio = int(words[8])
            
            (dataset[sectionId]).append(move_count)
            (dataset[sectionId]).append(work_count)
            (dataset[sectionId]).append(ratio)
            #printDataset(sectionId)
    print('loadDataFromFile ', filename)
    print('update Dataset [move_count,work_count,ratio]')        

filename = "input5_all.csv"

def loadCsvInput(path, filename):
    fullname = path+filename
    print('loadCsvInput ',fullname)
    return pd.read_csv(fullname)

def getPrevId(id,dataset):
    value = dataset[id]
    maxcount = -1
    maxid = id
    for prev in value[3]:
#        print(prev)
        if (dataset[prev] is None):
            print('dataset[prev] is None')
            continue
#        print(dataset[prev][5])
#        print(data[prev])
        if len(dataset[prev]) < 8:
            continue
        if (maxcount < dataset[prev][5]):
            maxcount = dataset[prev][5]
            maxid = prev

    if maxcount == -1:
        return None
            
    return maxid


def getNextId(id,dataset):
    value = dataset[id]
    maxcount = -1
    maxid = id
    for nxt in value[4]:
#        print(nxt)
        if (dataset[nxt] is None):
            print('dataset[prev] is None')
            continue
#        print(dataset[nxt][5])
#        print(data[nxt])
        if len(dataset[nxt]) < 8:
            continue
        if (maxcount < dataset[nxt][5]):
            maxcount = dataset[nxt][5]
            maxid = nxt
            
    if maxcount == -1:
        return None
    return maxid

def getX_3(id,data,dataset):
    if id not in data.columns:
        return None
    curS = data[id]
    prevId = getPrevId(id,dataset)
    if prevId not in data.columns:
        print(prevId, 'data[prevId] is None')
        return None
    prevS = data[prevId]
    nextId = getNextId(id,dataset)
    if nextId not in data.columns:
        print(nextId, 'data[prevId] is None')
        return None
    nextS = data[nextId]
    X = np.array(np.c_[prevS, curS, nextS])
#    print(prevId,prevId2,id,nextId,nextId2)
#    print(X.shape)
    return X

def getX(id,data,dataset):
    if id not in data.columns:
        return None
    curS = data[id]
    prevId = getPrevId(id,dataset)
    if prevId not in data.columns:
        print(prevId, 'data[prevId] is None')
        return None
    prevS = data[prevId]
    prevId2 = getPrevId(prevId,dataset)
    if prevId2 not in data.columns:
        print(prevId2, 'data[prevId] is None')
        return None
    prevS2 = data[prevId2]    
    nextId = getNextId(id,dataset)
    if nextId not in data.columns:
        print(nextId, 'data[prevId] is None')
        return None
    nextS = data[nextId]
    nextId2 = getNextId(nextId,dataset)
    if nextId2 not in data.columns:
        print(nextId2, 'data[prevId] is None')
        return None
    nextS2 = data[nextId2]
    X = np.array(np.c_[prevS2, prevS, curS, nextS, nextS2])
#    print(prevId,prevId2,id,nextId,nextId2)
#    print(X.shape)
    return X


def model(input_size, classes):
    X = tf.placeholder(tf.float32, [None, input_size])
    Y = tf.placeholder(tf.int32, [None, 1])
    Y_one_hot = tf.one_hot(Y, classes)
    Y_one_hot = tf.reshape(Y_one_hot, [-1, classes])

    W = tf.Variable(tf.random_normal([input_size, classes]), name='weight')
    b = tf.Variable(tf.random_normal([classes]), name='bias')

    logits = tf.matmul(X, W) + b
    hypothesis = tf.nn.softmax(logits)

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
    cost = tf.reduce_mean(cost_i)
    return hypothesis, cost

