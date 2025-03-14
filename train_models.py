import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # To suppress future warnings by xgboost dmatrix generator

import os
import time 
import pandas as pd
import numpy as np
import xgboost as xgb
from functools import wraps
import matplotlib.pyplot as plt
from copy import deepcopy as copy

from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init



##############################
# LOAD DATA
##############################

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

##############################
# XGBoost Related
##############################

def preprocess_for_xgboost(df, label_columns):
    """
    Preprocess the dataset for training an XGBoost model, including:
    - Encoding categorical features
    - Scaling features
    - Splitting the dataset into training, validation, and test sets
    - Handling missing values
    - Balancing the training set using SMOTE
    
    :param df: DataFrame containing the features and target columns
    :param label_columns: List or Set of column names for the target labels
    
    :return: dtrain, dvalid, dtest (XGBoost DMatrix objects for training, validation, and test sets)
    """
    
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
    assert isinstance(label_columns, (list, set)), "label_columns must be a list or set"
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoders for future use

    X = df.drop(columns=label_columns)  # Replace with actual target column
    y = df[list(label_columns)]
    scaler = StandardScaler()
    X.iloc[:, :] = scaler.fit_transform(X)


    smote = SMOTE()

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    imputer = SimpleImputer(strategy="mean")  # You can use "median" or "most_frequent"
    X_train = imputer.fit_transform(X_train)
    X_valid = imputer.transform(X_valid)
    X_test = imputer.transform(X_test)

    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    dtrain = xgb.DMatrix(X_train_balanced, label=y_train_balanced)
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test, label=y_test)

    return dtrain, dvalid, dtest

def evaluate_model_xgboost(model, dtest):
    """
    Evaluate and print the performance metrics of an XGBoost model on a test set.

    :param model: The trained XGBoost model.
    :param dtest: The test dataset in DMatrix format.

    :return: Accuracy score of the model on the test dataset.
    """
    
    # Assertions to check the types and validity of inputs
    assert isinstance(model, xgb.Booster), "Model must be a trained XGBoost Booster object"
    assert isinstance(dtest, xgb.DMatrix), "dtest must be an XGBoost DMatrix object"
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to binary values
    y_test = dtest.get_label()
    # Print Accuracy and Classification Report
    print(classification_report(y_test, y_pred))
    xgb.plot_importance(model)
    plt.show()
    return accuracy_score(y_test, y_pred)

def train_xgboost(dtrain, dvalid, dtest, verbose = True, *args, **kwargs):
    """
    Train an XGBoost model for binary classification.

    :param dtrain: The training dataset in DMatrix format.
    :param dvalid: The validation dataset in DMatrix format.
    :param dtest: The test dataset in DMatrix format.
    :param verbose: Bool value, If True, prints detailed progress of training and evaluation. Default is True.
    :param *args: Additional positional arguments to be passed to `xgb.train`.
    :param **kwargs: Additional keyword arguments to update the model's training parameters.

    :return: The trained XGBoost model.
    """
    assert isinstance(dtrain, xgb.DMatrix), "dtrain must be an xgb.DMatrix object"
    assert isinstance(dvalid, xgb.DMatrix), "dvalid must be an xgb.DMatrix object"
    assert isinstance(dtest, xgb.DMatrix), "dtest must be an xgb.DMatrix object"
    assert isinstance(verbose, bool), "Verbose must be bool"
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'logloss',  # Evaluation metric for binary classification
        'max_depth': 5,  # Maximum depth of trees
        'eta': 0.1,  # Learning rate
        'subsample': 0.2,  # Fraction of data used per tree
        'colsample_bytree': 0.8,  # Fraction of features used per tree
        'random_state': 42
    }

    params.update(**kwargs)
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    
    print("""
          
##################################################
*********    TRAINING XGBOOST MODEl     **********
##################################################
          
          """)
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,  # Stops if validation loss doesn't improve
        verbose_eval=50 if verbose else False
    )
    if verbose:
        evaluate_model_xgboost(model, dtest)

    return model


##############################
# Neural Network Related
##############################

class HeartDiseaseDataset(Dataset):
    """
    Custom PyTorch Dataset for heart disease data.
    
    This class takes a pandas DataFrame, processes it by handling categorical columns, and prepares the dataset
    for PyTorch models. It provides features and labels as tensors for training and evaluation.

    :params df: DataFrame containing features and labels.
    """

    def __init__(self, df):
        """
        Initializes the HeartDiseaseDataset by preprocessing the DataFrame.

        This includes:
        - Label encoding for the target column 'Heart Disease Status'
        - Label encoding for any categorical features
        - Conversion of features and labels to torch tensors

        :params df: DataFrame containing features and labels.
        """
        
        assert isinstance(df, pd.DataFrame)        
        assert 'Heart Disease Status' in df.columns
        
        # Convert 'Heart Disease Status' to numeric if it is not
        le = LabelEncoder()
        df['Heart Disease Status'] = le.fit_transform(df['Heart Disease Status'])
        self.labels = torch.tensor(pd.to_numeric(df['Heart Disease Status'], errors='coerce').values, dtype=torch.float32)
        
        # Separate features from the target column
        features = df.drop(columns=['Heart Disease Status'])
        
        # Handle categorical columns (assuming categorical columns are of type 'object' or 'category')
        categorical_columns = features.select_dtypes(include=['object', 'category']).columns
        label_encoders = {}
        
        # Apply label encoding to categorical features
        for col in categorical_columns:
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col])
            label_encoders[col] = le
        
        # Convert the DataFrame features into a torch tensor
        self.features = torch.tensor(features.values, dtype=torch.float32)
        
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: int
            The number of samples in the dataset.
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Fetches the features and label for a specific index.

        :param idx: The index of the data sample to retrieve.

        :return: A tuple containing:
                - The feature vector as a tensor.
                - The corresponding label as a tensor.
        """
        assert isinstance(idx, int), "Input idx must be an integer"
        return self.features[idx], self.labels[idx]


def preprocess_for_nn(df):    
    """
    Preprocess the DataFrame and create DataLoaders for training, validation, and test sets.

    :param df: A DataFrame containing the dataset. It is expected to have a 'Heart Disease Status' column as the target.

    :return: A tuple containing three DataLoader objects:
             - train_loader: DataLoader for the training set.
             - val_loader: DataLoader for the validation set.
             - test_loader: DataLoader for the test set.
    """

    assert isinstance(df, pd.DataFrame), "Input df must be a pandas DataFrame."
    assert 'Heart Disease Status' in df.columns,"'Heart Disease Status' column not found in the DataFrame."
    
    dataset = HeartDiseaseDataset(df)
    X_train, X_temp, y_train, y_temp = train_test_split(df.drop(columns=['Heart Disease Status']), 
                                                        df['Heart Disease Status'], 
                                                        test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create a DataLoader for batching
    train_df = X_train.copy()
    train_df['Heart Disease Status'] = y_train
    valid_df = X_valid.copy()
    valid_df['Heart Disease Status'] = y_valid
    test_df = X_test.copy()
    test_df['Heart Disease Status'] = y_test

    # Create Dataset objects
    train_dataset = HeartDiseaseDataset(train_df)
    valid_dataset = HeartDiseaseDataset(valid_df)
    test_dataset = HeartDiseaseDataset(test_df)

    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader


def get_predictions_from_nn(loader, model):
    """
    Returns the predictions and true labels from the neural network for the given data loader.

    :param loader: DataLoader
                   A PyTorch DataLoader object containing the dataset for which predictions will be made.
    :param model: nn.Module
                  A PyTorch model to make predictions.
    :param dtype: The dtype to which the input tensor should be cast.

    :return: tuple (np.ndarray, np.ndarray)
             - The predicted class labels as a NumPy array.
             - The true class labels as a NumPy array.
    """
    assert isinstance(loader, DataLoader), "loader must be a valid PyTorch DataLoader object"
    assert isinstance(model, (nn.Module, nn.Sequential)), "model must be an nn.Module or nn.Sequential"


    model_device = next(model.parameters()).device
    model.eval()  # Set model to evaluation mode

    all_preds = []
    all_labels = []
    
    score_array = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=model_device, dtype=torch.float32)
            y = y.to(device=model_device, dtype=torch.float32)
            scores = model(x)
            score_array.extend(scores.cpu().numpy())
            _, preds = scores.max(1)  # Get predicted class
            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_labels.extend(y.cpu().numpy())  # Store true labels
    scores = np.array(score_array)
    all_preds = (scores[:,0] < scores[:,1])+0.0
    all_labels = np.array(all_labels)
    return all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots the confusion matrix using the true and predicted labels.
    This function visualizes the confusion matrix as a heatmap, with labels annotated in each cell.


    :param y_true: List or array-like, containing the true labels of the dataset.
    :param y_pred: List or array-like, containing the predicted labels from the model.
    :param class_names: List of strings, containing the class labels for the dataset.

    :raises ValueError: If the number of class names does not match the confusion matrix dimensions.
    """
    
    # Ensure the inputs are valid
    assert isinstance(y_true, (list, np.ndarray)), "y_true must be a list or numpy array."
    assert isinstance(y_pred, (list, np.ndarray)), "y_pred must be a list or numpy array."
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."
    assert isinstance(class_names, list), "class_names must be a list of strings."
    assert len(class_names) > 0, "class_names should not be an empty list."

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape[0] != len(class_names) or cm.shape[1] != len(class_names):
        raise ValueError("The number of class names must match the dimensions of the confusion matrix.")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap="Blues")
    
    # Add color bar
    fig.colorbar(cax)
    
    # Add labels and title
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate the tick labels for better visibility
    plt.xticks(rotation=45)
    
    # Label the axes
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Annotate each cell with the numeric value
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    
    plt.show()



def init_kaiming_normal(m):
    """
    Initializes the weights of linear layers using Kaiming Normal initialization.

    :param m: Neural network layer 

    :return: None
    """
    if isinstance(m, nn.Linear):  # Apply to all Linear layers
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)  # Initialize biases to 0

def timeit(func):
    """
    Decorator to measure execution time of a function.

    :param func: The function whose execution time we want to measure.
    
    :return: A wrapper function that measures the time taken by the original function.
    """
    assert callable(func), f"Provided argument '{func}' must be function."

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Call the actual function
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.1f} seconds")
        return result
    return wrapper


def validation_accuracy(loader, model):
    """
    Calculates the accuracy of the model on the provided validation dataset.

    :param loader: DataLoader object containing the validation dataset.
    :param model: Model to evaluate.

    :return: A float representing the validation accuracy of the model on the given dataset.
    """
    assert isinstance(loader, DataLoader), "loader must be a valid PyTorch DataLoader object"
    assert isinstance(model, (nn.Module, nn.Sequential)), "model must be an nn.Module or nn.Sequential"
    all_preds, all_labels = get_predictions_from_nn(loader, model)
    if (all_preds==1).sum() < 10: 
        acc = 0
    else:
        acc = accuracy_score(all_labels, all_preds)
    return acc

@timeit
def train(model, optimizer, train_loader, val_loader, epochs=200, class_weights = [300, 1200], print_every = 50, test_every = 50, scheduler = None, apply_kaiming_init = True, verbose = True):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    :param model: A PyTorch model representing the model to train.
    :param optimizer: An optimizer used to train the model.
    :param train_loader: DataLoader object for the training dataset.
    :param val_loader: DataLoader object for the validation dataset.
    :param epochs: (Optional) Number of epochs to train the model. Default is 200.
    :param class_weights: (Optional) List of class weights for the loss function. Default is [300, 1200].
    :param print_every: (Optional) Int for how often to print the loss during training. Default is 50.
    :param test_every: (Optional) Int for how often to validate the model during training. Default is 50.
    :param scheduler: (Optional) Scheduler for the learning rate. Default is None.
    :param apply_kaiming_init: (Optional) Boolean value to apply Kaiming initialization to model weights. Default is True.
    :param verbose: (Optional) Boolean flag to control the verbosity of output. Default is True.

    :return: 
        - best_acc: Best validation accuracy obtained during training.
        - best_model_state: The state dictionary of the model with the best performance.
        - loss_list: List of total losses across epochs.
    """

    # Assertions for input validation
    assert isinstance(model, (nn.Module, nn.Sequential)), "model must be an nn.Module or nn.Sequential"
    assert isinstance(optimizer, optim.Optimizer), "optimizer must be an instance of optim.Optimizer"
    assert isinstance(train_loader, DataLoader), "train_loader must be a DataLoader object"
    assert isinstance(val_loader, DataLoader), "val_loader must be a DataLoader object"
    assert isinstance(epochs, int) and epochs > 0, "epochs must be a positive integer"
    assert isinstance(class_weights, list) and len(class_weights) == 2, "class_weights must be a list of two elements"
    assert all(isinstance(x, (int, float)) and x > 0 for x in class_weights), "Each element in class_weights must be a positive number"
    assert isinstance(print_every, int) and print_every > 0, "print_every must be a positive integer"
    assert isinstance(test_every, int) and test_every > 0, "test_every must be a positive integer"

    if apply_kaiming_init:
        model.apply(init_kaiming_normal)
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    class_weights = torch.tensor(class_weights, dtype= torch.float32)
    class_weights = class_weights.to(device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loss_list = []
    
    best_acc = 0
    best_model_state = None

    for e in range(epochs):
        total_loss = 0
        for t, (x, y) in enumerate(train_loader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = criterion(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_list.append(total_loss)
        
        if scheduler:
            scheduler.step()
        if e % test_every == 0:
            acc = validation_accuracy(val_loader, model)
            if acc > best_acc:
                best_acc = acc
                best_model_state = copy(model.state_dict())
        if verbose and e%print_every == 0:
        # print('\tEpoch %d, loss = %.4f' % (e, loss.item()))
            print('\tEpoch %d, loss = %.4f' % (e, total_loss))

    acc = validation_accuracy(val_loader, model)
    if acc > best_acc:
        best_acc = acc
        best_model_state = copy(model.state_dict())
    # print(scores)


    return best_acc, best_model_state, loss_list


def train_neural_network(
        train_loader,
        val_loader,
        num_epochs = 201,
        print_every=50,
        lr=5e-5,
        class_weights=[300,1200],
        verbose = True
    ):
    """
    Train a neural network, optimizing the hidden layer size and non-linearity function.

    :param train_loader: DataLoader for the training dataset.
    :param val_loader: DataLoader for the validation dataset.
    :param num_epochs: Number of epochs to train the model. Default is 201.
    :param print_every: Frequency (in epochs) to print the progress during training. Default is 50.
    :param lr: Learning rate for the optimizer. Default is 5e-5.
    :param class_weights: Weights for each class used in the loss function. Default is [300, 1200].
    :param verbose: Bool value, if True, prints classification report and confusion matrix. If False, prints the accuracy. Default is True.

    :return: The best trained model after optimization.
    """
    
    # Validating inputs
    assert isinstance(train_loader, torch.utils.data.DataLoader), "train_loader must be a DataLoader object"
    assert isinstance(val_loader, torch.utils.data.DataLoader), "val_loader must be a DataLoader object"
    assert isinstance(num_epochs, int) and num_epochs > 0, "num_epochs must be a positive integer"
    assert isinstance(print_every, int) and print_every > 0, "print_every must be a positive integer"
    assert isinstance(lr, (float, int)) and lr > 0, "lr must be a positive number"
    assert isinstance(class_weights, list) and len(class_weights) == 2, "class_weights must be a list of two elements"
    assert all(isinstance(x, (int, float)) and x > 0 for x in class_weights), "Each element in class_weights must be a positive number"

    
    # Initializing optimal parameter variables
    k_list = [16,32,64,96,128,160]
    val_acc_list = []
    loss_lists = {}
    best_k = None
    best_acc = 0
    best_model = None
    global_best_model_state = None

    print("""
          
##################################################
*********    TRAINING NEURAL NETWORK    **********
##################################################
          
          """)

    ## Train Part 1

    print(f"\nOptimizing Hidden Layer Size\n")
    for k in k_list:
        print("#####")
        print(f"Hidden Layer Size = {k}")
        model = nn.Sequential(nn.Linear(20, k),
                            nn.ReLU(),
                            nn.Linear(k, 2),
                            )
        
        optimizer = optim.Adam(model.parameters(), lr = lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs)
        acc, best_model_state, loss_list= train(model, optimizer, train_loader, val_loader, num_epochs, print_every=print_every,class_weights=class_weights, scheduler = scheduler, verbose=verbose)
        model.load_state_dict(best_model_state)
        loss_lists[k] = loss_list
        if acc > best_acc:
            best_k = k
            best_acc = acc 
            global_best_model_state = best_model_state 
            best_model = model

        val_acc_list.append(acc)
        y_pred, y_true = get_predictions_from_nn(val_loader, model)

        if verbose:
            print(classification_report(y_true, y_pred))
            plot_confusion_matrix(y_true, y_pred, class_names=["No Heart Disease", "Heart Disease"])
        else:
            print(f"Accuracy = {acc*100:.2f}%")
        print("#####")
    if verbose:
        plt.plot(k_list, val_acc_list)

    ## Train Part 2

    print(f"\nOptimizing Non-linearity\n")
    k = best_k
    non_linearity_list = [('ReLU',nn.ReLU), ('Sigmoid',nn.Sigmoid), ('Leaky ReLU',nn.LeakyReLU), ('Swish',nn.SiLU)]
    val_acc_list = []
    best_nonlinearity = None

    for string, nl in non_linearity_list:
        print("#####")
        print(f"Non-linearity = {string}")
        model = nn.Sequential(nn.Linear(20, k),
                            nl(),
                            nn.Linear(k, 2),
                            )
        optimizer = optim.Adam(model.parameters(), lr = lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        acc, best_model_state, loss_list = train(model, optimizer, train_loader, val_loader, num_epochs, print_every=print_every,class_weights=class_weights, scheduler = scheduler, verbose=verbose)
        model.load_state_dict(best_model_state)
        loss_lists[string] = loss_list
        if acc > best_acc:
            best_acc = acc 
            global_best_model_state = best_model_state 
            best_nonlinearity = string
            best_nl = nl
            best_model = model

        val_acc_list.append(acc)
        y_pred, y_true = get_predictions_from_nn(val_loader, model)

        if verbose:
            print(classification_report(y_true, y_pred))
            plot_confusion_matrix(y_true, y_pred, class_names=["No Heart Disease", "Heart Disease"])
        else:
            print(f"Accuracy = {acc*100:.2f}%")
        print("#####")
    if verbose:
        plt.bar([string for string, _ in non_linearity_list], val_acc_list)
        plt.title('Non linearity vs Accuracy')
        plt.xlabel('Non Linearity')
        plt.ylabel('Accuracy')

    return best_model

def load_model(filename, USE_GPU = True):
    """
    Loads a PyTorch model from a given file and sets it to evaluation mode.

    :param filename: String representing the file path of the saved PyTorch model

    :return: A PyTorch model loaded from the file
    """
    # Validate input
    assert isinstance(filename, str), "filename must be a string"
    assert os.path.isfile(filename), f"File not found: {filename}"
    assert isinstance(USE_GPU, bool), "USE_GPU must be a boolean value"

    model = torch.load(filename)
    assert isinstance(model, (nn.Module, nn.Sequential)), "model must be an nn.Module or nn.Sequential"

    model.to(device = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
    model.eval()
    return model

def predict(model, x):
    """
    Performs a forward pass using the given model and returns the predicted class 
    along with the class probabilities.

    :param model: A trained PyTorch model used for prediction
    :param x: A Tensor representing the input data

    :return: An integer representing the predicted class label (0 or 1) 
             and a list containing the class probabilities
    """
    assert isinstance(model, (nn.Module, nn.Sequential)), "model must be an nn.Module or nn.Sequential"
    assert isinstance(x, torch.Tensor), "x must be a torch.Tensor"
    assert any(p.requires_grad for p in model.parameters()), "model must have trainable parameters"

    model_device = next(model.parameters()).device
    x = x.to(device = model_device, dtype = torch.float32)
    y = model(x)
    pred_class = int(y[0]<y[1])
    probs = torch.exp(y)
    probs = probs/probs.sum()
    probs = probs.detach().cpu().tolist()
    return pred_class, probs





if __name__ == '__main__':

    # Configs
    data_file_path = './data/cleaned_heart_disease_majority.csv'
    save_file_path = './model_new.pth'
    verbose = True
    USE_GPU = True
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

    # Load data
    df = load_data(data_file_path)
    label_columns = {'Heart Disease Status'}

    # XGBoost 
    dtrain, dvalid, dtest = preprocess_for_xgboost(df, label_columns)
    xgboost_model = train_xgboost(dtrain, dvalid, dtest, verbose)

    # Neural Network
    # Training parameters and hyperparameters
    print_every = 50
    num_epochs = 201
    print_every = 50
    lr = 5e-5
    class_weights = [300,1200]
    verbose = True

    # Train neural network
    train_loader, val_loader, test_loader = preprocess_for_nn(df)
    model = train_neural_network(train_loader, val_loader, num_epochs, print_every, lr, class_weights, verbose)
    torch.save(model, save_file_path)


    






