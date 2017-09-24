import numpy as np
import matplotlib.pyplot as plt

def file2matrix(filename):
    fr = open(filename)
    dataOfile = fr.readlines()
    numOffile = len(dataOfile)
    numOffeature = len(dataOfile[0].split(',')) - 1
    inputData = np.zeros((numOffile, numOffeature))
    outputData = np.zeros(numOffile)
    index = 0
    for line in dataOfile:
        line = line.strip()
        listOfline = line.split(',')
        inputData[index,:] = listOfline[0:-1]
        outputData[index] = listOfline[-1]
        index += 1
    return inputData, outputData

def featurescaling(inputData):
    m, n = inputData.shape
    meanData = np.mean(inputData, axis = 0) #compute mean
    maxData = np.max(inputData, axis = 0)   #compute max
    normData = (inputData - meanData)/maxData
    return normData

def normalequation(inputdata, outputdata):
    m, n = inputdata.shape
    inputdata = np.column_stack((np.ones(m),inputdata))
    theta = np.dot(inputdata.T, inputdata)
    theta = np.linalg.pinv(theta)
    theta = np.dot(theta, inputdata.T)
    theta = np.dot(theta, outputdata.T)
    return theta

def predict(inputdata, theta):
    m, n = inputdata.shape
    inputdata = np.column_stack((np.ones(m), inputdata))
    predictRes = np.dot(inputdata, theta.T)
    return predictRes

if __name__ == "__main__":
    inputdata, outputdata = file2matrix(r'E:\test\untitled\data.txt')
    theta = normalequation(inputdata[0:800,:], outputdata[0:800])
    predictRes = predict(inputdata[800:,:],theta)
    np.savetxt('res.txt', predictRes, fmt='%.2f')


