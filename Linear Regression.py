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

def gradientdescent(inputdata, outputdata, alpha):
    m, n = inputdata.shape
    inputdata = np.column_stack((np.ones(m), inputdata)) #add x0 = 1 to inputdata matrix
    theta = np.zeros(n + 1)
    J = []
    for i in range(500):
        hypothesis = np.dot(inputdata, theta)
        loss = hypothesis - outputdata
        gradient = np.dot(inputdata.T, loss) / m
        #print(gradient)
        #gradient = np.dot(inputdata.T, np.dot(inputdata, theta) - outputdata)
        theta = theta - alpha * gradient
        temp = np.dot(inputdata, theta) - outputdata    #X*theta - y
        tempJ = 1/2 * np.dot(temp.T, temp)/m
        J.append(tempJ)
    plt.plot(range(500), J, 'r')
    plt.show()

    return theta


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
    inputdata, outputdata = file2matrix(r'E:\test\untitled\train.txt')
    inputdata_test, outputdata_test = file2matrix(r'E:\test\untitled\test.txt')

    # theta = normalequation(inputdata, outputdata)
    # predictRes = predict(inputdata_test,theta)
    # np.savetxt('res.txt', predictRes, fmt='%.2f')

    normdata = featurescaling(inputdata)
    theta1 = gradientdescent(normdata, outputdata, 0.03)
    predictRes = predict(inputdata_test,theta1)
    np.savetxt('res_gd.txt', predictRes, fmt='%.2f')

