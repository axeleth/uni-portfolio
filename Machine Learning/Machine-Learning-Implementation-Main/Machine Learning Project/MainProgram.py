import pandas as pd
import numpy as np

from kNN import kNN
from NeuralNetwork import NeuralNetwork
from DataFunctions import DataFunctions

def Main():
    np.random.seed(1)

    ##########################################################################################
    # Params
    #
    #
    # Dataset: path to the dataset
    # Datasplit: how are we going to divide the training, validation and testdata
    # K: how many neighbours we are going to use for our classification in kNN
    ##########################################################################################

    PathToDataSet: str = "Machine Learning Project/milknew.csv"
    DataSplit: dict = {"training": 0.6, "validation": 0.2, "test": 0.2 }
    K: int = 5

    TrainData, ValidationData, TestData = DataFunctions.InitializeDataSet(PathToDataSet, DataSplit)

    RunNeuralNetwork(TrainData, TestData)
    RunKNearestNeighbours(TrainData, TestData, K)




def RunNeuralNetwork(TrainData: pd.DataFrame, TestData: pd.DataFrame):
    # Create a neural network instance with 2 hidden layers, the final layer is the output layer
    networkSize = [5,3,1,4,3,1]
    learningRate = 0.01
    dropoutProbability = 0.3
    network = NeuralNetwork(TrainData, networkSize, learningRate, dropoutProbability)

    network.TrainNetwork()

    predicted, actual = network.TestNetwork(TestData)

    ##########################################################################################
    # Network evaluation
    ##########################################################################################
    errors = 0
    n = len(predicted)
    for i in range(n):
        classification = round(predicted[i] * 2) / 2
        if classification != actual[i]:
            errors += 1

    error = (actual - predicted) ** 2
    print("Predicted: ", predicted)
    print("Actual: ", actual)
    print("Error: ", error)
    print("error average: ", np.average(error))
    print("weights: ", network.Weights)

    print("")
    print("")
    print("Number of values in dataset: " + str(n))
    print("Number of errors: " + str(errors))
    print("Percentage errors: " + str(errors/n))

def RunKNearestNeighbours(TrainData, TestData, K):
    knn = kNN(K)
    results = knn.RunKNN(TrainData, TestData)

    print(results)

    

if __name__ == "__main__":
    Main()