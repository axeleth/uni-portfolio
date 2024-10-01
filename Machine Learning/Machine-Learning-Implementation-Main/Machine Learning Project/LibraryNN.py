import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.neural_network import MLPClassifier

DataSplit: dict = {"training": 0.6, "validation": 0.2, "test": 0.2 }

def LoadDataset():
        CsvData = pd.read_csv('Machine-Learning-Implementation/Machine Learning Project/milknew.csv') #
        Length: int = len(CsvData) #needed to split the data into parts

        CsvData.iloc[:,-1] = CsvData.iloc[:,-1].map({'low':0.0, 'medium':0.5, 'high':1.0})

        #Split data into train, validation, test
        TrainData = CsvData[0: int(Length * DataSplit["training"])] 
        ValidationData = CsvData[int(Length * DataSplit["training"]): int(Length * DataSplit["training"]) + int(Length * DataSplit["validation"])]
        TestData = CsvData[int(Length * DataSplit["training"]) + int(Length * DataSplit["validation"]): Length] 

        for i in range(6):
            TrainData = TrainData.append(TrainData.iloc[1:,:])
        
        #shuffle train data
        TrainData = TrainData.sample(frac=1).reset_index(drop=True)

        return TrainData, ValidationData, TestData


def NormalizeData(DataSet):
        n: int = len(DataSet.columns) #Number of columns

        #the last column contains the grade, which doesn't need to be normalized as it is a classification and not a number
        #loop over all columns and normalize them
        for i in range(n - 1): 
            LowerBound: int = min(DataSet.iloc[:,i]) #Column min
            UpperBound: int = max(DataSet.iloc[:,i]) #Column max

            DataSet.iloc[:,i] = (DataSet.iloc[:,i] - LowerBound) / (UpperBound - LowerBound) #Using this function, the column will be normalized to be between 0 and 1
        
    
        return DataSet

TrainData, ValidationData, TestData = LoadDataset()


def tfNN(TrainData):
        model = tf.keras.Sequential([
                tf.keras.layers.Dense(3,input_shape=(7,), activation='sigmoid'),
                tf.keras.layers.Dense(5, activation='sigmoid'),
                tf.keras.layers.Dense(7, activation='sigmoid'),
                tf.keras.layers.Dense(5, activation='sigmoid'),
                tf.keras.layers.Dense(3, activation='softmax')
                ])

        model.compile(optimizer='adam',
                        loss=keras.losses.MeanAbsoluteError(),
                        metrics=['accuracy'])
        
        model.fit(
                TrainData.iloc[:,:-1].values, 
                TrainData.iloc[:,-1].values, 
                batch_size=64,
                epochs=10)
        
        model.evaluate(TestData.iloc[:,:-1].values, TestData.iloc[:,-1].values, verbose=2)
        
tfNN(TrainData)

