"""
Some functions to preprocess the data
InitializeDataSet: convert the dataset into a dataframe
Normalizedata: make sure the values in every columns are between 0 and 1
"""

import pandas as pd

class DataFunctions:
    @staticmethod
    def InitializeDataSet(PathToDataSet: str, DataSplit: dict):
        CsvData = pd.read_csv(PathToDataSet)
        Length: int = len(CsvData) 

        #Split data into train, validation, test
        TrainData = CsvData[0: int(Length * DataSplit["training"])] 
        ValidationData = CsvData[int(Length * DataSplit["training"]): int(Length * DataSplit["training"]) + int(Length * DataSplit["validation"])]
        TestData = CsvData[int(Length * DataSplit["training"]) + int(Length * DataSplit["validation"]): Length] 

        return DataFunctions.NormalizeData(TrainData), DataFunctions.NormalizeData(ValidationData), DataFunctions.NormalizeData(TestData)

    @staticmethod
    def NormalizeData(DataSet: pd.DataFrame):
        n: int = len(DataSet.columns) #Number of columns

        # the last column contains the grade, which doesn't need to be normalized as it is a classification and not a number
        # loop over all columns and normalize them using the minimum and maximum column value
        # and then make sure every value is between 0 and 1 
        for i in range(n - 1): 
            lowerBound: int = min(DataSet.iloc[:,i]) 
            upperBound: int = max(DataSet.iloc[:,i])

            DataSet.iloc[:,i] = (DataSet.iloc[:,i] - lowerBound) / (upperBound - lowerBound)
    
        return DataSet