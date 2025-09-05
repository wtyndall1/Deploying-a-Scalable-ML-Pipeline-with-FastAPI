import pickle
# TODO: add necessary import - done
import numpy as np
import pandas as pd
from typing import Any, Tuple
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # TODO: implement the function - done
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(
    y: np.ndarray,
    preds: np.ndarray
) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: Any, X: np.ndarray) -> np.ndarray:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # TODO: implement the function - done
    return model.predict(X)

def save_model(model, path: str) -> None:
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    # TODO: implement the function - done
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path: str):
    """ Loads pickle file from `path` and returns it."""
    # TODO: implement the function - done
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
    data: pd.DataFrame,
    column_name: str,
    slice_value,
    categorical_features: list,
    label: str,
    encoder,
    lb,
    model: Any) -> Tuple[float, float, float]:
    """ Computes the model metrics on a slice of the data
    specified by a column name and

    Processes the data using one hot encoding for the
    categorical features and a
    label binarizer for the labels. This can be used
    in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
        Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical
        features (default=[])
    label : str
        Name of the label column in `X`. If None, then
        an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # TODO: implement the function - done
    df_slice = data[data[column_name] == slice_value]

    if df_slice.shape[0] == 0:
        return 0.0, 0.0, 0.0
    
    X_slice, y_slice, _, _ = process_data(
        df_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,

    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta