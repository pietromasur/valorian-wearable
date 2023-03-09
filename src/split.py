from sklearn.model_selection import train_test_split
import pandas as pd

def custom_train_test_split(df, train_size = 0.7, val_size = 0.15, test_size=0.15, random_state = 42):
  """
    Splits the input dataframe into custom train, validation and test sets.

    Args:
        df (pandas.DataFrame): The input dataframe.
        train_size (float, optional): The proportion of the dataset to include in the train set. Defaults to 0.7.
        val_size (float, optional): The proportion of the dataset to include in the validation set. Defaults to 0.15.
        test_size (float, optional): The proportion of the dataset to include in the test set. Defaults to 0.15.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.

    Returns:
        tuple: A tuple containing the train, validation and test sets for the input dataframe.
    """
  
  if (train_size + val_size + test_size) != 1:
    raise ValueError("Size of train, valid and test sets must add up to 1.")

  t_map = {'Standing':0, 'Sitting':1, 'Downstairs':2, 'Walking':3, 'Jogging':4, 'Upstairs':5}
  features_train, target_train, features_test, target_test, features_val, target_val = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
  for activity in df.atividade.unique():
    ac_df = df[df.atividade==activity]
    features = ac_df[['a_x', 'a_y', 'a_z']]
    target = ac_df.atividade
    target = target.map(t_map)
    
    X_train, X_val_test, y_train, y_val_test = train_test_split(features, target, test_size=train_size-(test_size+val_size), random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_size / (val_size + test_size), random_state=random_state)

    features_train = pd.concat([features_train, X_train], axis=0, ignore_index=True)
    target_train = pd.concat([target_train, y_train], axis = 0, ignore_index=True)
    features_val = pd.concat([features_val, X_val], axis=0, ignore_index=True)
    target_val = pd.concat([target_val, y_val], axis=0, ignore_index=True)
    features_test = pd.concat([features_test, X_test], axis = 0, ignore_index=True)
    target_test = pd.concat([target_test, y_test], axis = 0, ignore_index=True)
    

  return features_train, target_train, features_val, target_val, features_test, target_test