from typing import Tuple, List, Optional
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_data(
    raw_df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and validation sets using stratification.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Original dataset.
    target_col : str
        Name of target column.
    test_size : float
        Fraction of data for validation.
    random_state : int
        Random seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and validation dataframes.
    """
    train_df, val_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[target_col]
    )
    return train_df, val_df


def separate_inputs_targets(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features and target column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Inputs dataframe and target series.
    """
    inputs = df.drop(columns=[target_col]).copy()
    targets = df[target_col].copy()
    return inputs, targets


def drop_identifier_columns(
    df: pd.DataFrame,
    columns_to_drop: List[str]
) -> pd.DataFrame:
    """
    Drop identifier columns from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns_to_drop : List[str]
        List of columns to remove.

    Returns
    -------
    pd.DataFrame
        Dataframe without identifier columns.
    """
    return df.drop(columns=columns_to_drop)


def encode_categorical_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    Perform One-Hot Encoding for categorical features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training inputs.
    val_df : pd.DataFrame
        Validation inputs.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]
        Encoded train dataframe, encoded validation dataframe, fitted encoder.
    """
    categorical_cols = train_df.select_dtypes(include="object").columns.tolist()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    train_encoded = pd.DataFrame(
        encoder.transform(train_df[categorical_cols]),
        columns=encoded_cols,
        index=train_df.index
    )

    val_encoded = pd.DataFrame(
        encoder.transform(val_df[categorical_cols]),
        columns=encoded_cols,
        index=val_df.index
    )

    train_df = train_df.drop(columns=categorical_cols)
    val_df = val_df.drop(columns=categorical_cols)

    train_df = pd.concat([train_df, train_encoded], axis=1)
    val_df = pd.concat([val_df, val_encoded], axis=1)

    return train_df, val_df, encoder


def scale_numeric_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numeric features with more than two unique values using StandardScaler.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training inputs.
    val_df : pd.DataFrame
        Validation inputs.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]
        Scaled train dataframe, scaled validation dataframe, fitted scaler.
    """
    numeric_cols = train_df.select_dtypes(include="number").columns

    scaled_cols = [
        col for col in numeric_cols
        if train_df[col].nunique() > 2
    ]

    scaler = StandardScaler()
    scaler.fit(train_df[scaled_cols])

    train_df[scaled_cols] = scaler.transform(train_df[scaled_cols])
    val_df[scaled_cols] = scaler.transform(val_df[scaled_cols])

    return train_df, val_df, scaler


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = False
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    List[str],
    Optional[StandardScaler],
    OneHotEncoder
]:
    """
    Full preprocessing pipeline:
    - split data
    - separate inputs and targets
    - drop identifier columns
    - encode categorical features
    - optionally scale numeric features

    Parameters
    ----------
    raw_df : pd.DataFrame
        Original dataset.
    scaler_numeric : bool
        Whether to scale numeric features.

    Returns
    -------
    Tuple containing:
        X_train : pd.DataFrame
        y_train : pd.Series
        X_val : pd.DataFrame
        y_val : pd.Series
        input_cols : List[str]
            Original input column names.
        scaler : Optional[StandardScaler]
        encoder : OneHotEncoder
    """
    target_col = "Exited"
    input_cols = raw_df.columns[:-1].tolist()

    train_df, val_df = split_data(raw_df, target_col)

    X_train, y_train = separate_inputs_targets(train_df, target_col)
    X_val, y_val = separate_inputs_targets(val_df, target_col)

    id_cols = ['id', 'CustomerId', 'Surname']
    X_train = drop_identifier_columns(X_train, id_cols)
    X_val = drop_identifier_columns(X_val, id_cols)

    X_train, X_val, encoder = encode_categorical_features(X_train, X_val)

    scaler = None
    if scaler_numeric:
        X_train, X_val, scaler = scale_numeric_features(X_train, X_val)

    return X_train, y_train, X_val, y_val, {
        "encoder": encoder,
        "scaler": scaler,
        "input_cols": input_cols,
        "id_cols": ['id', 'CustomerId', 'Surname']
    }



def preprocess_new_data(new_df, config):
    df = new_df.copy()

    encoder = config["encoder"]
    scaler = config["scaler"]
    id_cols = config["id_cols"]

    # Drop id columns
    df = df.drop(columns=id_cols, errors="ignore")

    # Encode
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    encoded_df = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoded_cols,
        index=df.index
    )

    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    # Scale
    if scaler is not None:
        numeric_cols = df.select_dtypes(include="number").columns
        scaled_cols = [col for col in numeric_cols if df[col].nunique() > 2]
        df[scaled_cols] = scaler.transform(df[scaled_cols])

    return df