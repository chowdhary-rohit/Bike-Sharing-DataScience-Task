import pandas as pd

class DataIngestion:
    def __init__(self, path):
        """
        Initialization
            :param self: 
            :param path: path to the imput csv file
        """   
        self.path = path
        self.data = pd.read_csv(self.path)

    def getdata(self):
        """
        read all the data in a pandas dataframe
            :param self: 
        """   
        
        return self.data