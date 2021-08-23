"""
Data preparation methods for continuous variables.
"""
import pandas as pd


def fill_numeric(dataframe: pd.DataFrame, col: str, fill_type: str = 'median') -> pd.DataFrame:
    """Fills missing values in numeric column specified.
    Args:
        dataframe: DataFrame to fill
        col: Column in DataFrame to fill
        fill_type: How to fill the data. Supported types: "mean", "median", "-1"
    Returns:
        A DataFrame with numeric_col filled.
    """
    if fill_type == 'median':
        fill_value = dataframe[col].median()  # type: float
    elif fill_type == 'mean':
        fill_value = dataframe[col].mean()
    elif fill_type == '-1':
        fill_value = -1
    else:
        raise NotImplementedError('Valid fill_type options are "mean", "median", "-1')

    dataframe.loc[dataframe[col].isnull(), col] = fill_value
    return dataframe
