import numpy as np
import sys

class Network:
    def __init__(self, weightsHidden ,weightsOutput):
        self.HiddenLayerW = weightsHidden
        self.OutputLayerW = weightsOutput

    def Forward(self, inputs):
        HiddenLayer = np.dot(inputs, self.HiddenLayerW)
        ActivationHiddenLayer = self.Sigmoid(HiddenLayer)
        OutputLayer = np.dot(ActivationHiddenLayer, self.OutputLayerW)
        output = self.Sigmoid(OutputLayer)
        return 1 if output > 0.5 else 0

    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

#This function read the data from the file and init it to a struction
def LoadData(fileData):
    with open(fileData, 'r') as file:
        lines = file.readlines()

    firstLine = lines[0].split(",")
    inputSize, HiddenSize,OutputSize=int(firstLine[0]),int(firstLine[1]),int(firstLine[2])

    HiddenW = lines[1].strip()[1:-1]
    OutputW = lines[2].strip()[1:-1]
    weightsHidden = np.array([float(w) for w in HiddenW.split(',')]).reshape(inputSize, HiddenSize)
    weightsOutput = np.array([float(w) for w in OutputW.split(',')]).reshape(HiddenSize, OutputSize)
    
    return weightsHidden ,weightsOutput

#This function run the network and get the label in a file
def GetLabelAndSaveDate(network, fileData, fileOutput):
    with open(fileData, 'r') as file:
        lines = file.readlines()
    outputs = [str(network.Forward(np.array([int(ch) for ch in line.strip() if ch.isdigit()], dtype=float))) + '\n' for line in lines]
    with open(fileOutput, 'w') as file:
        file.writelines(outputs)




if __name__ == '__main__':

    args = sys.argv[1:]
    if len(args) > 0:
        Went0File = args[0]
        if len(args) > 1:
            Went0File = args[1]
    else:
        Went0File="wnet0.txt"
        Datafile="test"

    WeightsHidden ,weightsOutput=LoadData(Went0File)
    GetLabelAndSaveDate(Network(WeightsHidden ,weightsOutput), Datafile, 'testnet0.txt')