import pandas as pd
from kmodes.kmodes import KModes
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

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

class KModesImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that imputes missing categorical values using k-modes clustering.
    """
    def __init__(self, n_clusters=3, missing_placeholder="missing"):
        self.n_clusters = n_clusters
        self.missing_placeholder = missing_placeholder
        
    def fit(self, X, y=None):
        # Ensure X is a DataFrame and store column names
        self.X_columns = X.columns
        # Fill missing values with the placeholder
        X_filled = X.fillna(self.missing_placeholder)
        
        # Apply k-modes clustering on the completed data
        self.km = KModes(n_clusters=self.n_clusters, init='Huang', n_init=5, verbose=0)
        self.clusters_ = self.km.fit_predict(X_filled)
        
        # Attach the cluster labels for mode computation
        X_filled = X_filled.copy()
        X_filled['cluster'] = self.clusters_
        
        # Compute the mode for each column within each cluster
        self.imputed_values_ = {}
        for cluster in range(self.n_clusters):
            cluster_data = X_filled[X_filled['cluster'] == cluster]
            mode_values = {}
            for col in self.X_columns:
                # Only consider non-placeholder values when computing the mode
                non_missing = cluster_data[cluster_data[col] != self.missing_placeholder][col]
                if not non_missing.empty:
                    mode_values[col] = non_missing.mode().iloc[0]
                else:
                    # If all values are missing in the cluster, fall back to the placeholder
                    mode_values[col] = self.missing_placeholder
            self.imputed_values_[cluster] = mode_values
        return self
    
    def transform(self, X):
        X_transformed = X.copy().fillna(self.missing_placeholder)
        # Assign clusters using the fitted k-modes model
        clusters = self.km.predict(X_transformed)
        X_transformed['cluster'] = clusters
        # Impute missing values using the cluster modes
        for col in self.X_columns:
            X_transformed[col] = X_transformed.apply(
                lambda row: self.imputed_values_[row['cluster']][col] 
                            if row[col] == self.missing_placeholder else row[col],
                axis=1
            )
        # Drop the temporary cluster column
        X_transformed = X_transformed.drop(columns=['cluster'])
        return X_transformed
    

def clean_data_Kmodes(data_frame):
    """ 
    Clean the data up from any missing values using KNN Imputer on
    numerical columns and Simple Imputer on non-numerical columns.
    Produces a cleaned data frame without missing entries. 

    :param data_frame: The data frame to clean up

    :return: The cleaned data frame
    """
    assert(isinstance(data_frame, pd.DataFrame)), "The input must be DataFrame object"

    # df = load_data("heart_disease.csv")
    # KNNImputer only works on numerical data
    # Apply to numerical columns with missing values
    numerical_cols = data_frame.select_dtypes(include=['number']).columns
    knn_imputer = KNNImputer(n_neighbors=5)
    data_frame[numerical_cols] = knn_imputer.fit_transform(data_frame[numerical_cols])

    # Apply k-modes for non-numerical columns
    non_numerical_cols = data_frame.select_dtypes(exclude=['number']).columns
    kmodes_imputer = KModesImputer(n_clusters=3, missing_placeholder="missing")
    data_frame[non_numerical_cols] = kmodes_imputer.fit_transform(data_frame[non_numerical_cols])
    
    # Apply SimpleImputer for non-numerical columns
    # non_numerical_cols = data_frame.select_dtypes(exclude=['number']).columns
    # mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    # data_frame[non_numerical_cols] = mode_imputer.fit_transform(data_frame[non_numerical_cols])

    return data_frame

def main():
    df = load_data("heart_disease.csv")
    cleaned_df = clean_data_Kmodes(df)
    return cleaned_df
    # cleaned_df.to_csv("cleaned_heart_disease_kmodes.csv", index=False) 