import pandas as pd
import IPython
print (IPython.sys_info())
class DataIngestion:
    def __init__(self, path):
        """Initialize the path to the input data file
        
        Arguments:
            path {[type]} -- location of the input csc file
        """
        self.path = path
        self.data = pd.read_csv(self.path)

    def getdata(self):
        """Read all the data in a pandas dataframe
        
        Returns:
            [type] -- the complete data set
        """
        return self.data