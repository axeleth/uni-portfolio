import math
import pandas as pd
import numpy as np

class kNN:
    def __init__(self, k: int):
        self.K = k

    def GetNearestNeighbours(self, data: pd.DataFrame, node):
        # calculation using the apply function is much quicker than using iloc directly
        # due to optimization in the pd apply function
        distances = data.iloc[:,:-1].apply(lambda row: self.Distance(row, node), axis=1)
        neighbourList = pd.concat([data.iloc[:,:-1], distances], axis=1)
        
        # Rename column to distance
        neighbourList = neighbourList.rename(columns={0: 'Distance'}).sort_values(by='Distance', ascending=True)
        
        return neighbourList.iloc[:self.K, :]

    def ClassifyNewNode(self, nearestNeighbours: pd.DataFrame, node):        
        gradeCount = nearestNeighbours.iloc[:,-2].value_counts().to_dict()

        newNode = node.copy()
        newNode['Distance'] = sorted(gradeCount)[0]

        return newNode

    def RunKNN(self, trainData: pd.DataFrame, testData: pd.DataFrame):
        results = testData
        for i in range(testData.shape[0]):
            nearestNeighboursToNode = self.GetNearestNeighbours(trainData, testData.iloc[i,:])
            results.iloc[i,:] = self.ClassifyNewNode(nearestNeighboursToNode, testData.iloc[i,:-1])

        return results

    @staticmethod
    def Distance(node1, node2) -> float:
        x: float = np.sum([(node1[i] - node2[i]) ** 2 for i in range((len(node1) - 1))])        

        return np.sqrt(x)