from dataclasses import dataclass
import numpy as np
import pandas as pd
import math

class NeuralNetwork:
    def __init__(self, data: np.array, networkSize: list, learningRate: int, dropoutProbability: int):
        np.random.seed(1)
        """
        ExpectedOutput: value that is changed every time a value is computed to match the expected output
        Neurons: output neurons we used to calculate the output
        Weights
        Output: A single value representing the class

        Learning rate: multiplier for backpropagation to prevent overfitting for every calculation we do
        NetworkSize: Array with the number of neurons in every layer
        Expectedoutpus: All expected outputs from the data
        DropoutProbability: The probability that a random weight is going to be missing, used for training in an attempt 
        to prevent overfitting
        """

        # Initialization for class variables we are going to mutate later
        self.ExpectedOutput = math.inf
        self.Neurons = []
        self.Weights = []
        self.Output = math.inf

        # Class initialization
        self.Input = self.ConvertInputToNpArray(data)
        self.LearningRate = learningRate
        self.NetworkSize = networkSize
        self.DropoutProbability = dropoutProbability
        self.ExpectedOutputs = self.ClassesToNumericValues(data)

        # Set the start weights to random values corresponding to a normal distribution
        self.GetStartWeights()

    def ConvertInputToNpArray(self, data) -> np.ndarray:
        # get all the rows and all the columns except the classification column as a numpy array
        return np.array(data.iloc[:, :-1].values) 

    def ClassesToNumericValues(self, data) -> np.array:
        grades: np.ndarray = np.array(data.iloc[:, -1].values)
        
        gradeNumberPairs = {'low': 0, 'medium': 0.5, 'high': 1}
        for i in range(len(grades)):
            grades[i] = gradeNumberPairs[grades[i]]
        
        return grades
    
    def Sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def SigmoidDerivative(self, x: np.ndarray) -> np.array:
        sig_x = self.Sigmoid(x)
        return sig_x * (1 - sig_x)   

    def GetLossDerivative(self) -> float:
        classification = round(self.Output[0] * 2) / 2     
        return 2 * (self.ExpectedOutput - classification )
    
    def GetLayerStartWeights(self, InputLayerSize: int, OutputLayerSize: int):
        return np.random.rand(InputLayerSize, OutputLayerSize)

    def GetStartWeights(self):
        weights = []
        for i in range(len(self.NetworkSize)):
            if i == 0:
                weights.append(self.GetLayerStartWeights(self.Input.shape[1], self.NetworkSize[i]))
            else:
                weights.append(self.GetLayerStartWeights(self.NetworkSize[i - 1], self.NetworkSize[i]))
        
        self.Weights = weights

    def ComputeNeuralNetwork(self, row: int):
        neurons = [self.Input[row]]
        for i in range(len(self.Weights)):
            neurons.append(self.Sigmoid(np.dot(neurons[i], self.Weights[i])))
        
        self.Output = neurons[-1]
        self.ExpectedOutput = self.ExpectedOutputs[row]
        self.Neurons = neurons


    def BackPropagate(self):
        deltas = []
        for i in range(len(self.Neurons) - 1, -1, -1):
            if i == len(self.Neurons) - 1:
                deltas.append(self.GetLossDerivative() * self.SigmoidDerivative(self.Neurons[i]))
            else:
                deltas.append(np.dot(deltas[-1], self.Weights[i].T) * self.SigmoidDerivative(self.Neurons[i]))
        
        deltas.reverse()
        for i in range(len(self.Weights)):
            self.Weights[i] += self.LearningRate * self.Neurons[i].T.dot(deltas[i])

    def Dropout(self, x: np.ndarray) -> np.ndarray:
        # Set random weights to zero, according to the dropout probability
        mask = np.random.binomial(1, 1 - self.DropoutProbability, size=x.shape) / (1 - self.DropoutProbability)
        return x * mask

    def TrainNetwork(self):
        originalWeights = self.Weights

        for _ in range (1000):
            dropoutWeights = [self.Dropout(layer) for layer in originalWeights.copy()]

            for i in range(len(self.Input)):
                self.Weights = dropoutWeights
                self.ComputeNeuralNetwork(i)

                self.BackPropagate()


    def TestNetwork(self, testData: pd.DataFrame):
        self.Input = self.ConvertInputToNpArray(testData)

        expectedOutput = self.ClassesToNumericValues(testData)
        predictedOutput = np.array([])

        n = len(self.Input)
        for i in range(n):
            self.ComputeNeuralNetwork(i)
            predictedOutput = np.append(predictedOutput, self.Output)
    
        return predictedOutput, expectedOutput


