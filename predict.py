import numpy as np
import pandas as pd

#Visualization
import seaborn as sns 
import matplotlib.pyplot as plt

#sklearn model selection
from sklearn import model_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import metrics

#sklearn regression models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_ingestion import DataIngestion

from sklearn.model_selection import RandomizedSearchCV

import random
import os.path
from prettytable import PrettyTable
from pprint import pprint
import pickle

# matplotlib inline 
sns.set(color_codes=True)

class RandomForest():
    ''' Random forest data model wrapper to provide additional methods for the Bike Sharing Dataset.
    
    '''

    def __init__(self, data_path='/rawdata/hour.csv'):
        '''Initialize the random forest model.
        
        Keyword Arguments:
            data_path {str} -- Path to the Bike Sharing Dataset. (default: {'/rawdata/hour.csv'})            
        '''
        
        # Make results reproducible
        random.seed(100)

        # Load data form bike sharing csv
        self.data = {}
        dataingestion = DataIngestion(data_path)
        self.data = dataingestion.getdata()

        # Define feature and target variables
        self.features= ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
        self.target = ['cnt']

        self.X=np.array(self.data[self.features])
        self.y=np.array(self.data[self.target]).ravel()

        #Splitting the data into training and test data sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2 , random_state = 1)

        # Define model 
        self.model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                    max_features='auto', max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=1, min_samples_split=4,
                    min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
                    oob_score=False, random_state=100, verbose=0, warm_start=False)
        
        
    def savePickleModel(self, file="pickle_model.pkl"):
        """Generate a pickle file to store the model on the disk
        
        Keyword Arguments:
            file {str} -- Path for the model (default: {"pickle_model.pkl"})
        
        Returns:
            boolean -- success of data storage
        """
        
        success = False
        if self.model is not None:
            with open(file, 'wb') as file:  
                pickle.dump(self.model, file)
            success = True
        return success

        
    def loadPickleModel(self, file="pickle_model.pkl"):
        """Load the model from generated pickle file
        
        Keyword Arguments:
            file {str} -- Path for the model (default: {"pickle_model.pkl"})
        
        Returns:
            boolean -- success of data loading
        """
        # Load the model from generated pickle file
        success = False
        if os.path.exists(file):
            with open(file, 'rb') as file:  
                self.model = pickle.load(file)            
            success = True
        return success
        
        
    def kFoldCrossvalidation(self):      
        """
        Runs the ten-fold cross-validation on the dataset
        """

        table = PrettyTable()
        table.field_names = ["Model", "R² score", "Mean Absolute Error", "Mean Absolute Deviation"]

        kfold = model_selection.KFold(n_splits=10, random_state=21)
        cv_results = cross_validate(self.model, self.X_train, self.y_train, cv=kfold, scoring=('r2', 'neg_mean_absolute_error'),
                                return_train_score=True)

        #Defining the metrics for evaluation                        
        accuracy = format(np.mean(cv_results['test_r2']), '.2f') 
        mae = format(np.negative(np.mean(cv_results['test_neg_mean_absolute_error'])), '.2f')
        mad = format(np.negative(np.std(cv_results['test_neg_mean_absolute_error'])), '.2f')

        table.add_row(["RandomForestRegressor", accuracy, mae, mad])
        print (table)

    def train(self):  
        """Train the random forest model on the training data set.

        """        
        self.model.fit(self.X_train, self.y_train)
    
    def test(self):
        """ Evaluate the performance of Random Forest on the test data set.
        
        Returns:
            [type] -- [dictionary with the test results]
        """

        # Check model loaded
        if self.model is None:
            print("Please load or train a model before!")
            return   
            
        pred = self.model.predict(self.X_test)

        r2_score(self.y_test, pred)
        mae = mean_absolute_error(self.y_test, pred) 

        return {'r2_score':r2_score, 'mae':mae}
    
    def hyperparameter(self):      
        """ Finding the best parameters for random forest model to optimize performance
        
        Returns:
            [type] -- [dictionary with the set of parameters]
        """   

        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        rf = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        return rf_random.get_params()
    
    def featureImportance(self):
        """
        Plotting the feature importances of the trained random forest regressor model 
        """     
        # Check model loaded
        if self.model is None:
            print("Please load or train a model before!")
            return
           
        kfold = model_selection.KFold(n_splits=10, random_state=21)
        for train, _ in kfold.split(self.X, self.y):
            self.model.fit(self.X[train, :], self.y[train])
            importances = self.model.feature_importances_
            std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
            indices = np.argsort(importances)[::-1]
            # Print the feature ranking
            print("Feature ranking:")
           
            for f in range(self.X_test.shape[1]):
                print("%d. feature %s (%f)" % (f + 1, self.features[indices[f]], importances[indices[f]]))
            # Plot the feature importances of the forest
            plt.figure(figsize=(18,5))
            plt.title("Feature importances")
            plt.bar(range(self.X_test.shape[1]), importances[indices], color="cornflowerblue", yerr=std[indices], align="center")
            plt.xticks(range(self.X_test.shape[1]), [self.features[i] for i in indices])
            plt.xlim([-1, self.X_test.shape[1]])
            
            print('Would you like to continue to calculate feature importance for the next fold? (yes/no)')
            user_input = input()
            if (user_input == "yes" or user_input == "Yes" or user_input == "YES"):
                continue
            else:
                break
        plt.show()      

if __name__ == "__main__":
    
    model = RandomForest()
    #model.savePickleModel()
    #model.loadPickleModel()
    #model.train()
    #model.test()

    model.kFoldCrossvalidation()
    #model.featureImportance()
    #model.hyperparameter()