import pandas as pd
import plotly.express as px
import seaborn as sns
import os


def correct_txt_line_ends(file_path, corrected_file_name = 'coleta_formated.csv'):
    """
    Function to correct line endings of a text file.

    Args:
        file_path (str): Path to the text file to be corrected.
        corrected_file_name (str): Name of the corrected file.

    Returns:
        None
    """
    with open(file_path, 'r') as file :
        filedata = file.read()
        # Replace the target string
        filedata = filedata.replace(',;', ';')
        filedata = filedata.replace(';', os.linesep)
    # Write the file out again
    with open(corrected_file_name, 'w') as file:
        file.write(filedata)

def clean_and_resample(file_path, resampled_df_path = 'resampled_data.csv', **kwargs):
    """
    Function to clean and resample accelerometer data from a CSV file.
    Args:
        file_path (str): Path to the CSV file containing the accelerometer data.
        resampled_df_path (str): Name of the file to save the resampled data.

    Returns:
        None
    """
    df = _clean_csv_file(file_path)
    _resample_data(df, resampled_df_path = resampled_df_path)

def _clean_csv_file(df_path):
    """
    Function to clean accelerometer data from a CSV file.
    Args:
        df_path (str): Path to the CSV file containing the accelerometer data.

    Returns:
        pandas.DataFrame: A cleaned DataFrame.
    """
    header = ['individuo', 'atividade', 'timestamp', 'a_x', 'a_y', 'a_z']
    df = pd.read_csv(df_path, names = header)
    df = df.drop_duplicates(subset = ['individuo', 'timestamp'])
    df = df.dropna()
    return df

def _resample_data(df, window_size = 3000, overlap = 0.75, resampled_df_path = 'resampled_data.csv'):
    """
    Function to resample accelerometer data from a DataFrame.

    Args:
        df (pandas.DataFrame): A DataFrame containing the accelerometer data.
        window_size (int): The length of the rolling window (in milliseconds).
        overlap (float): The amount of overlap between adjacent windows (as a fraction of window_size).
        resampled_df_path (str): Name of the file to save the resampled data.

    Returns:
        None
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
    df = df.set_index('timestamp')
    step = int(window_size*(1-overlap))
    resampled_df = pd.DataFrame(columns = df.columns)
    individuals = df['individuo'].unique()
    for i in individuals:
        #Selectings individual
        individual_df = df[df['individuo']==i]
        #Sorting by time
        individual_df = individual_df.sort_values(by='timestamp')
        # Select only numerical values
        df_numeric = individual_df.select_dtypes(include=['datetime','float64', 'int64'])
        # Calculating rolling window
        df_windowed = df_numeric.rolling(str(window_size)+'ms').mean()
        # print(df_windowed)
        # Combine rolling data with non-numeric data
        df_windowed = pd.concat([df_windowed, individual_df.select_dtypes(exclude=['float64', 'int64'])], axis=1)

        # Step to make the overlap
        df_windowed_shifted = df_windowed.shift(step)
        df_windowed_shifted = df_windowed_shifted.dropna()

        resampled_df = pd.concat([resampled_df, df_windowed_shifted], axis=0)
    
    resampled_df.reset_index(inplace = True)
    resampled_df.columns = ['timestamp', 'individuo', 'atividade', 'a_x', 'a_y', 'a_z']
    resampled_df.to_csv(resampled_df_path, index=False)
