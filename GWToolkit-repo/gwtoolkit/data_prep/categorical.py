"""
Data preparation methods for categorical variables.
"""
import pandas as pd
import numpy as np


def lowercase_string(string: str) -> str:
    """Returns a lowercased string
    Args:
        string: String to lowercase
    Returns:
        String in lowercase
    """
    return string.lower()


def lowercase_column(dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
    """Lowercases a column in a dataframe
    Args:
        dataframe: DataFrame to lowercase
        col: Column in DataFrame to lowercase
    Returns:
        A DataFrame with column lowercased
    """
    dataframe[col] = dataframe[col].apply(lowercase_string)
    return dataframe


def extract_title(dataframe: pd.DataFrame, col: str, replace_dict: dict = None,
                  title_col: str = 'title') -> pd.DataFrame:
    """Extracts titles into a new title column
    Args:
        dataframe: DataFrame to extract titles from
        col: Column in DataFrame to extract titles from
        replace_dict (Optional): Optional dictionary to map titles
        title_col: Name of new column containing extracted titles
    Returns:
        A DataFrame with an additional column of extracted titles
    """
    dataframe[title_col] = dataframe[col].str.extract(r' ([A-Za-z]+)\.', expand=False)

    if replace_dict:
        dataframe[title_col] = np.where(dataframe[title_col].isin(replace_dict.keys()),
                                 dataframe[title_col].map(replace_dict),
                                 dataframe[title_col])

    return dataframe
