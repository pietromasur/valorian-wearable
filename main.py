
"""
This script is the entry point of the machine learning pipeline. It contains a set of functions to load, clean and preprocess data, split it into training, validation and test sets, and finally train and evaluate a Random Forest and an XGBoost classifiers.

It imports the following modules:
- src.clean: a module with functions for cleaning and preprocessing data
- src.split: a module with functions for splitting data into training, validation and test sets
- src.train_and_evaluate: a module with functions for training and evaluating machine learning models

This script uses the following functions:
- correct_txt_line_ends: a function from src.clean module that corrects the line ends in a text file and save it as a csv file
- clean_and_resample: a function from src.clean module that reads a csv file, preprocesses and resamples data, and save it as a csv file
- custom_train_test_split: a function from src.split module that splits data into training, validation and test sets
- train_and_evaluate: a function from src.train_and_evaluate module that trains and evaluates a machine learning model and saves the results

This script executes the following steps:
- Corrects the line ends in a text file and saves it as a csv file using the correct_txt_line_ends function
- Cleans, preprocesses and resamples data and saves it as a csv file using the clean_and_resample function
- Reads the resampled data file and splits it into training, validation and test sets using the custom_train_test_split function
- Trains and evaluates a Random Forest classifier using the train_and_evaluate function with the 'rf' model_name parameter and saves the results in the 'results/rf' directory
- Trains and evaluates an XGBoost classifier using the train_and_evaluate function with the 'xgb' model_name parameter and saves the results in the 'results/xgb' directory
"""

from src.clean import *
from src.split import *
from src.train_and_evaluate import *

def __main__():
    correct_txt_line_ends('data/coleta.txt', 'data/coleta_formated.csv')
    clean_and_resample('data/coleta_formated.csv', 'data/resampled_data.csv')
    
    df = pd.read_csv('data/resampled_data.csv')
    features_train, target_train, features_val, target_val, features_test, target_test = custom_train_test_split(df)
    
    train_and_evaluate('rf','results/rf', features_train, target_train, features_val, target_val, features_test, target_test)
    train_and_evaluate('xgb','results/xgb', features_train, target_train, features_val, target_val, features_test, target_test)

if __name__ == '__main__':
    __main__()