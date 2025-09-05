import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    performance_on_categorical_slice,
)

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

#creating dataframe to use for tests
def _tiny_df() -> pd.DataFrame:
    """
    # Small but realistic DataFrame covering both classes and categories.
    """
    rows = [
        [39, "Private", "Bachelors", "Never-married", "Tech-support",
         "Not-in-family", "White", "Male", "United-States", 13, 77516,
         0, 0, 40, "<=50K"],
        [50, "Self-emp-not-inc", "Bachelors", "Married-civ-spouse",
         "Exec-managerial", "Husband", "White", "Male", "United-States",
         13, 83311, 0, 0, 13, ">50K"],
        [38, "Private", "HS-grad", "Divorced", "Handlers-cleaners",
         "Not-in-family", "Black", "Female", "United-States", 9, 215646,
         0, 0, 40, "<=50K"],
        [53, "Private", "11th", "Married-civ-spouse",
         "Machine-op-inspct", "Husband", "Black", "Male",
         "United-States", 7, 234721, 0, 0, 40, "<=50K"],
        [28, "Private", "Bachelors", "Married-civ-spouse",
         "Prof-specialty", "Wife", "White", "Female", "United-States",
         13, 338409, 0, 0, 40, ">50K"],
        [37, "Private", "Masters", "Never-married", "Prof-specialty",
         "Not-in-family", "Asian-Pac-Islander", "Male", "India", 14,
         284582, 0, 0, 50, ">50K"],
        [45, "State-gov", "Masters", "Divorced", "Prof-specialty",
         "Unmarried", "White", "Female", "United-States", 14, 141297,
         0, 0, 38, "<=50K"],
        [34, "Private", "Some-college", "Separated", "Sales",
         "Unmarried", "White", "Male", "Canada", 10, 121313, 0, 0, 45,
         ">50K"],
    ]
    cols = [
        "age", "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country", "education-num",
        "fnlgt", "capital-gain", "capital-loss", "hours-per-week",
        "salary",
    ]
    return pd.DataFrame(rows, columns=cols)

# TODO: implement the first test. Change the function name and input as needed
def test_train_and_inference_shape():
    """
    # Trains a model on a tiny dataset and verifies inference returns a 1-D
    array whose length matches X_test rows.
    """
    # Your code here
    df = _tiny_df()
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df["salary"]
    )

    X_train, y_train, enc, lb = process_data(
        train_df, categorical_features=CAT_FEATURES, label="salary",
        training=True
    )
    X_test, y_test, _, _ = process_data(
        test_df, categorical_features=CAT_FEATURES, label="salary",
        training=False, encoder=enc, lb=lb
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert preds.ndim == 1
    assert preds.shape[0] == X_test.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

# TODO: implement the second test. Change the function name and input as needed
def test_metrics_computation_known_case():
    """
    # Checks precision, recall, and F1 against a simple known example.
    y_true = [0,1,1,0]; y_pred = [0,1,0,0] -> TP=1, FP=0, FN=1
    Precision=1.0, Recall=0.5, F1=0.6666
    """
    # Your code here
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    p, r, f1 = compute_model_metrics(y_true, y_pred)

    assert pytest.approx(p, rel=1e-6) == 1.0
    assert pytest.approx(r, rel=1e-6) == 0.5
    assert pytest.approx(f1, rel=1e-6) == (2 * (1 * 0.5) / (1 + 0.5))


# TODO: implement the third test. Change the function name and input as needed
def test_slice_performance_existing_and_empty():
    """
    # Computes slice metrics for an existing slice (should return floats in
    [0,1]) and for an empty slice (should return zeros).
    """
    # Your code here
    df = _tiny_df()
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=0, stratify=df["salary"]
    )

    X_train, y_train, enc, lb = process_data(
        train_df, categorical_features=CAT_FEATURES, label="salary",
        training=True
    )
    model = train_model(X_train, y_train)

    # Existing slice
    col = "sex"
    val = test_df[col].iloc[0]
    p, r, f1 = performance_on_categorical_slice(
        data=test_df, column_name=col, slice_value=val,
        categorical_features=CAT_FEATURES, label="salary",
        encoder=enc, lb=lb, model=model
    )
    for m in (p, r, f1):
        assert isinstance(m, float)
        assert 0.0 <= m <= 1.0

    # Empty slice
    p0, r0, f10 = performance_on_categorical_slice(
        data=test_df, column_name="native-country", slice_value="Neverland",
        categorical_features=CAT_FEATURES, label="salary",
        encoder=enc, lb=lb, model=model
    )
    assert (p0, r0, f10) == (0.0, 0.0, 0.0)
