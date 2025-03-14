import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from enum import Enum

def load_data(file_path: str):
    """
    Returns a DataFrame object of the csv file passed in.

    :param file_path: String of the file path to load in

    :return: A DataFrame object of the csv data
    """
    assert(isinstance(file_path, str)), "File path must be a valid path"
    # file_path = "./data/heart_disease.csv"
    df = pd.read_csv(file_path)
    return df

def get_data_info(data_frame):
    """
    View the structure of the data frame

    :param data_frame: The data frame to get the structure of
    """
    assert(isinstance(data_frame, pd.DataFrame)), "The input must be DataFrame object"
    print("Summary of Dataset:")
    data_frame.info()
    print("Get missing count")
    data_frame.isnull().sum() 

def get_num_rows(data_frame: pd.DataFrame):
    """
    Get number of rows of the data frame

    :param data_frame: The data frame to get the number of rows
    """
    assert(isinstance(data_frame, pd.DataFrame)), "The input must be DataFrame object"
    return data_frame.shape[0]

def get_num_cols(data_frame: pd.DataFrame):
    """
    Get number of columns of the data frame

    :param data_frame: The data frame to get the number of columns
    """
    assert(isinstance(data_frame, pd.DataFrame)), "The input must be DataFrame object"
    return data_frame.shape[1]

def classify_non_numerical_columns(data_frame):
    """
    Classification model that predicts the output of non-numerical
    data in the data frame for missing entries.

    :param data_Frame: The data_frame.

    :return: A new data frame with all of the classified columns.
    """
    assert(isinstance(data_frame, pd.DataFrame)), "The input must be DataFrame object"
    non_numeric_cols = data_frame.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        label_encoder = LabelEncoder()
        data_frame[col] = label_encoder.fit_transform(data_frame[col])
    return data_frame

def drop_random_rows_no_heart_disease(data_frame: pd.DataFrame):
    """
    Drop random rows from the given pandas DataFrame.
    Used to drop random entries that do not have heart disease.
    This is to balance the dataset. There are about 2000
    with heart disease and 8000 without heart disease.
    This is skewing the data visualizations and prediction model
    to just 80% not having heart disease.

    :param df: The pandas DataFrame if the number of
    no heart disease and heart disease do not match.
    """
    assert(isinstance(data_frame, pd.DataFrame) and data_frame.shape[0] > 0), "Argument must be a valid non-empty pandas DataFrame"
    assert("Heart Disease Status" in data_frame.columns), "There must be a Heart Disease Status column in the pandas DataFrame"
    ser = data_frame["Heart Disease Status"].value_counts()
    num_heart_disease, num_no_heart_disease = ser["Yes"], ser["No"]
    assert(num_heart_disease > 0 and num_no_heart_disease > 0), "Number of Yes and No Heart Disease Status in the DataFrame should be greater than 0. If not, not a good dataset"
    if num_heart_disease > num_no_heart_disease:
        num_to_drop = num_heart_disease - num_no_heart_disease
        indices_to_drop = data_frame[data_frame["Heart Disease Status"].str.lower() == "yes"].sample(num_to_drop).index
        data_frame = data_frame.drop(indices_to_drop)
    elif num_no_heart_disease > num_heart_disease:
        num_to_drop = num_no_heart_disease - num_heart_disease
        indices_to_drop = data_frame[data_frame["Heart Disease Status"].str.lower() == "no"].sample(num_to_drop).index
        data_frame = data_frame.drop(indices_to_drop)
    return data_frame

class ImputerMethod(Enum):
    KNN = "KNN"
    SIMPLE = "Simple"
    DROP = "Drop"

def clean_data(data_frame: pd.DataFrame, method: ImputerMethod):
    """ 
    Clean the data up from any missing values (if any) by just dropping
    these rows or by using KNN Imputer on numerical columns and
    Simple Imputer on non-numerical columns. Produces a cleaned
    data frame without missing entries. 

    :param data_frame: The data frame to clean up

    :param method: The method of cleaning the data. Can either drop all rows with missing entries,
    or use KNN Imputer on numerical columns and simple Imputer on non-numericla columns.
    Method is of type ImputerMethod enum.

    :return: The cleaned data frame if there are any rows that have missing entries
    """
    assert(isinstance(data_frame, pd.DataFrame)), "The input must be DataFrame object"
    assert(isinstance(method, ImputerMethod)), "The input must be an imputer method either KNN or SIMPLE"

    if data_frame.isnull().any(axis=1).sum():
        if method == ImputerMethod.KNN:
            # KNNImputer only works on numerical data
            # Apply to numerical columns with missing values
            numerical_cols = data_frame.select_dtypes(include=['number']).columns
            knn_imputer = KNNImputer(n_neighbors=5)
            data_frame[numerical_cols] = knn_imputer.fit_transform(data_frame[numerical_cols])

            # Apply SimpleImputer for non-numerical columns
            non_numerical_cols = data_frame.select_dtypes(exclude=['number']).columns
            mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            data_frame[non_numerical_cols] = mode_imputer.fit_transform(data_frame[non_numerical_cols])
        elif method == ImputerMethod.DROP:
            data_frame = data_frame.dropna()
    #data_frame = drop_random_rows_no_heart_disease(data_frame)
    return data_frame

def run():
    """
    Main function to run the complete data processing/cleaning.

    :return: df_cleaned_drop, df_cleaned_knn which are the
    cleaned up datasets with their respective methods.
    """
    df = load_data("../data/heart_disease.csv")
    print(f"Num rows before cleaning: {get_num_rows(df)}")
    print(f"Num of cols before cleaning: {get_num_cols(df)}\n")

    df_cleaned_drop = clean_data(df, ImputerMethod.DROP)
    print(f"Num of rows after cleaning with DROP: {get_num_rows(df_cleaned_drop)}")
    print(f"Num of cols after cleaning with DROP: {get_num_cols(df_cleaned_drop)}")
    duplicate_counts = df_cleaned_drop.duplicated().sum()
    print(f"Duplicate rows in cleaned DROP: {duplicate_counts}\n")

    df_cleaned_knn = clean_data(df, ImputerMethod.KNN)
    print(f"Num of rows after cleaning with KNN: {get_num_rows(df_cleaned_knn)}")
    print(f"Num of cols after cleaning with KNN: {get_num_cols(df_cleaned_knn)}")
    duplicate_counts = df_cleaned_knn.duplicated().sum()
    print(f"Duplicate rows in cleaned KNN: {duplicate_counts}\n")
    
    return df_cleaned_drop, df_cleaned_knn


def main():
    """
    Main function for cleaning the dataset and balancing the rows between
    "No Heart Disease" and "Heart Disease" Status to reduce bias in the 
    machine learning model as well as the analytics. Adds new columns
    that store the numerical data of "Cholesterol Level", "BMI" and
    "Blood Pressure" as categorical data for easy data visualization. 

    Use this for your final cleaned dataset for visualization.
    """

    df_cleaned_drop, df_cleaned_knn = run()

    cleaned_drop_series = df_cleaned_drop["Gender"].value_counts()
    print(f"{cleaned_drop_series}\n")
    num_men_drop, num_women_drop = cleaned_drop_series
    print(f"Number of men in clean with DROP: {num_men_drop}")
    print(f"Number of women in clean with DROP: {num_women_drop}\n")

    cleaned_knn_series = df_cleaned_knn["Gender"].value_counts()
    print(f"{cleaned_knn_series}\n")
    num_men_knn, num_women_knn = cleaned_knn_series
    print(f"Number of men in clean with KNN: {num_men_knn}")
    print(f"Number of women in clean with KNN: {num_women_knn}")

    # Use the KNN_Imputer Method and get equal distrubtion of heart disease
    # status to prevent bias in dataset (previous was bias towards no heart disease)
    cleaned_data = drop_random_rows_no_heart_disease(df_cleaned_knn)

    print(cleaned_data["Heart Disease Status"].value_counts())
    # Create a new column that quantizes the Cholesterol Levels
    cleaned_data["Cholesterol Category"] = pd.cut(
        cleaned_data["Cholesterol Level"],
        bins=[0, 199, 239, float("inf")],  
        labels=["Normal", "Elevated", "High"],  
        right=True  
    )
    cleaned_data['Cholesterol Category'] = pd.Categorical(cleaned_data['Cholesterol Category'], categories=["Normal", "Elevated", "High"], ordered=True)

    # Create a new column that quantizes the BMI ranges
    cleaned_data["BMI Category"] = pd.cut(
        cleaned_data["BMI"],
        bins=[0, 18.5, 25, 30, float("inf")],  
        labels=["Underweight", "Normal", "Overweight", "Obese"],  
        right=True  # Include right edge in the bin
    )
    cleaned_data['BMI Category'] = pd.Categorical(cleaned_data['BMI Category'], categories=["Underweight", "Normal", "Overweight", "Obese"], ordered=True)

    # Create a new column that quantizes the Blood Pressure ranges
    cleaned_data["Blood Pressure Category"] = pd.cut(
        cleaned_data["Blood Pressure"],
        bins=[0, 120, 129, 139, float("inf")], 
        labels=["Normal", "Elevated", "High", "Very High"],  
        right=True  
    )
    cleaned_data['Blood Pressure Category'] = pd.Categorical(cleaned_data['Blood Pressure Category'], categories=["Normal", "Elevated", "High", "Very High"], ordered=True)

    cleaned_data.to_csv("../data/equal_distribution_hds.csv")

    return cleaned_data