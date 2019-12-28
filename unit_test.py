import unittest
from sklearn.ensemble import RandomForestRegressor

from predict import RandomForest


class TestModel(unittest.TestCase):
    ''' Test the methods of the random forest model wrapper.
    
    Arguments:
        unittest -- Python Unit Test Framework
    '''

    def setUp(self):
        ''' Test runner setup.
        '''
        
        print('setUp') 
        self.rf = RandomForest()     
        self.rf.model_file = 'pickle_model.pkl'

    def testDataVariables(self):
        ''' Test whether feature and target variables are defined in the model.
        '''
        
        self.assertGreater(len(self.rf.features), 0)
        self.assertGreater(len(self.rf.target), 0)
 
    
    def testDataLoading(self):
        ''' Test whether data is loaded or not.
        '''
        
        self.assertGreater(len(self.rf.data), 0)
    
    def testSavePickleModel(self):
        ''' Test the save method for a blank model.
        '''

        self.rf.model = RandomForestRegressor()
        self.assertTrue(self.rf.savePickleModel())
    
    def testSaveNonePickleModel(self):
        ''' Test the save method for a non existing model.
        '''

        self.rf.model = None
        self.assertFalse(self.rf.savePickleModel())

    def testLoadPickleModel(self):
        ''' Test the load model for a previous saved model.
        '''

        self.rf.savePickleModel()
        self.assertTrue(self.rf.loadPickleModel())

    def testLoadNonePickleModel(self):
        ''' Test the load method for a non-existing model path.
        '''

        self.rf.model_file = 'non-existing/model_path'
        self.assertFalse(self.rf.loadPickleModel())
    
if __name__ == '__main__':
    #run tests 
    unittest.main()