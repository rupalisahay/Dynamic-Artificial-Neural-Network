

import pandas as pd
import numpy as np
import math
from math import exp
from operator import mul



#remove nulls from dataset
def removeNulls(dataFrame):
    updated_dataFrame = dataFrame
    for i in range(len(dataFrame)):
        for j in range(len(dataFrame.iloc[i, :])):
            if(dataFrame.iloc[i,j] == '?'):
                updated_dataFrame = updated_dataFrame.drop(dataFrame.index[i])
                break

    return updated_dataFrame


#normalize the dataset
def meanNormalize(array):
    mean = np.average(array)
    std_dev = np.std(array)
    normalizedArray = (array - mean)/std_dev
    return normalizedArray


#enter input path of the dataset
url = input("Enter the URL of the dataset:")
output_path = input("Enter the output path for processed data:")
dataFrame = pd.read_csv(url, skipinitialspace=True, na_values='.', header = None)



dataFrame_without_nulls = removeNulls(dataFrame)



normalised_dataFrame = dataFrame_without_nulls
for i in range(len(dataFrame_without_nulls.columns)):
    data_type = dataFrame_without_nulls.dtypes[i]
    #print("TYPE:", data_type)
    if(data_type == np.int64 or data_type == np.float64):
        normalised_dataFrame[i] = meanNormalize(dataFrame_without_nulls[i])

        
        

preprocessed_dataFrame = normalised_dataFrame
for j in range(len(preprocessed_dataFrame.columns)):
    data_type = preprocessed_dataFrame.dtypes[j]
    if(data_type == np.object):
        list_unique = preprocessed_dataFrame[j].unique().tolist()
        
        for i in range(len(preprocessed_dataFrame)):
            value = preprocessed_dataFrame.iloc[i, j]
            label = list_unique.index(value)
            preprocessed_dataFrame.iloc[i, j] = label
        



j = len(preprocessed_dataFrame.columns)-1
preprocessed_dataFrame[j]
list_unique = preprocessed_dataFrame[j].unique().tolist()

a = list(list_unique)
n = len(a)
interval = 1/(n-1)
b = a
value = 0
for i in range(len(a)):
    b[i] = value
    value = value + interval

b = np.array(b)


for i in range(len(preprocessed_dataFrame)):
    value = preprocessed_dataFrame.iloc[i, j]
    label = b[list_unique.index(value)]
    preprocessed_dataFrame.iloc[i, j] = label




normalised_dataFrame.to_csv(output_path, sep=',',index= None, header=None)


#sigmoid funtion
def sigmoidActivationFunction(netValue):
    if(netValue > 20):
        netValue = 20
    elif(netValue<-20):
        netValue = -20
        
    return 1.0 / (1.0 + exp(-netValue))

#train neural network
def trainNeuralNetwork(maximumIterations, dataFrame, allWeights, numberOfNodes, learningRate):
    for n in range(maximumIterations):    
        for i in range(len(dataFrame)):
            inputArray = dataFrame.iloc[i,:-1]
            inputArray = list(inputArray)
            inputArray.insert(0, 1)
            inputArray = np.array(inputArray)
        
            targetOutput = dataFrame.iloc[i,-1]
            sigmoidList, output = forwardPropagation(allWeights, inputArray, numberOfNodes)

            allWeights = backPropagationFunction(allWeights, inputArray, sigmoidList, numberOfNodes, learningRate, targetOutput)
            
    return allWeights


def calculateWeights(inputNodeCount, hiddenLayersCount, hiddenNodesCount):
    hiddenNodesCount.insert(0, inputNodeCount)
    numberOfNodes = hiddenNodesCount

    allWeights = list()
    
    for i in range(hiddenLayersCount):
        j = i + 1
        W = np.matrix(np.random.uniform(-1,1, size=(numberOfNodes[j]-1, numberOfNodes[i])))
        allWeights.append(W.tolist())

    i = i + 1
    j = i + 1
    W = np.matrix(np.random.uniform(-1,1, size=(outputNodesCount, numberOfNodes[i])))
    allWeights.append(W.tolist())
    return allWeights, numberOfNodes



def findAccuracyOfNetwork(dataFrame, allWeights, numberOfNodes):
    
    j = len(dataFrame.columns)-1
    dataFrame[j]
    list_unique = dataFrame[j].unique().tolist()
    list_unique.sort()
    num_corrects = 0
    for i in range(len(dataFrame)):
        inputArray = dataFrame.iloc[i,:-1]
        inputArray = list(inputArray)
        inputArray.insert(0, 1)
        inputArray = np.array(inputArray)
        
        targetOutput = dataFrame.iloc[i,-1]

        sigmoidList, output = forwardPropagation(allWeights, inputArray, numberOfNodes)
        
        label  = findLabel(list_unique, output[0])
        if(label == targetOutput):
           num_corrects = num_corrects+1
    accuracy = num_corrects/len(dataFrame)
    return accuracy



#forward propagation function

def forwardPropagation(allWeights, inputArray, numberOfNodes):

    sigmoidList = list()
    for i in range(len(numberOfNodes)):
        temp = 0
        sigmoid_at_level = list()
        sigmoid_at_level.append(1)
        #print("\n\nAt layer", i)
        if(i==0):
            vector = inputArray
        else:
            vector
            
        temp = np.array(vector)*allWeights[i]
        
        netValue = list()
        for j in range(len(temp)):
            netValue.append(sum(temp[j]))
        
        
        for j in range(len(netValue)):
                sigmoid_at_level.append(sigmoidActivationFunction(netValue[j]))
        sigmoidList.append(sigmoid_at_level)
        vector = sigmoid_at_level
    
    #The output
    output = sigmoid_at_level[1:]
    return sigmoidList, output


def findLabel(array,value):
    index = np.searchsorted(array, value, side="left")
    if index > 0 and (index == len(array) or math.fabs(value - array[index-1]) < math.fabs(value - array[index])):
        return array[index-1]
    else:
        return array[index]


#back propagation function

def backPropagationFunction(allWeights, inputArray, sigmoidList, numberOfNodes, learningRate, targetOutput):
    #print("Back Prop")
    output = sigmoidList[-1][1]
    x = len(numberOfNodes)
    
    list_delta = list()
    
    for i in range(len(numberOfNodes), 0, -1):
        delta_at_level = list()
        delta_weights_level = 0
        if(i==len(numberOfNodes)):
            temp = (targetOutput - output)*output*(1-output)
            delta_at_level.insert(0, temp)
            list_delta.insert(0, delta_at_level)
        else:
            delta_at_level = np.array(sigmoidList[i-1]) * (1 - np.array(sigmoidList[i-1])) * np.array(list_delta[x-i-1]) * np.array(allWeights[i])
            
            delta_weights_level = learningRate * list_delta[x-i-1][0] * np.array(sigmoidList[i-1])
            list_delta.insert(0, delta_at_level.tolist())
        
            mytemp = allWeights[i][0] + delta_weights_level
            allWeights[i][0] = list(mytemp)
            
           

    
    delta_weights_level = learningRate * np.matrix(list_delta[0][0][1:]).T * np.matrix(inputArray)
    delta_weights_level = delta_weights_level.tolist()

    
    a = np.matrix(delta_weights_level)
    b = np.matrix(allWeights[0])
    c = np.add(a,b)
    temp = c.tolist()
    allWeights[0] = temp
    
    return allWeights





#input from user

path_input_dataset = input("Enter the input Dataset to form the Neural Network : ")
dataFrame = pd.read_csv(path_input_dataset, header=None)
dataFrame = dataFrame.sample(frac=1)
training_percent = int(input("Enter the percent of training data to be used : "))

maximumIterations = int(input("Enter the Maximum iterations :"))
learningRate = 0.9
count_features = len(dataFrame.columns)-1+1 # -1 for the class label, +1 for bias


hiddenLayersCount = int(input("Enter the total number of hidden layers:"))

hiddenNodesCount = list()
for i in range(0, hiddenLayersCount):
    value = int(input("Enter the nodes in each of the hidden layers : "))
    hiddenNodesCount.append(value+1)
    
 

inputNodeCount = count_features
count_weight_matrices = hiddenLayersCount + 1
outputNodesCount = 1

num_examples = len(dataFrame)
index = int(training_percent * num_examples / 100)
training_dataset = dataFrame[0:index]
testing_dataset = dataFrame[index:]

allWeights, numberOfNodes = calculateWeights(inputNodeCount, hiddenLayersCount, hiddenNodesCount)




allWeights = trainNeuralNetwork(maximumIterations, training_dataset, allWeights, numberOfNodes, learningRate)





training_accuracy = findAccuracyOfNetwork(training_dataset, allWeights, numberOfNodes)




testing_accuracy = findAccuracyOfNetwork(testing_dataset, allWeights, numberOfNodes)





training_error = 1 - training_accuracy
testing_error = 1 - testing_accuracy



num_layers = len(allWeights)

for i in range(num_layers):
    print("\n\nLayer ", i)
    allWeights_layer = allWeights[i]
 
    for k in range(len(allWeights_layer[0])):
        print("\n\tNeuron", k, "weights:\n")
        for j in range(len(allWeights_layer)):
            print("\t\t",allWeights_layer[j][k])




print("Total Training Error:", training_error*100)
print("Total Testing Error:", testing_error*100)






