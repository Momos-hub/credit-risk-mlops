"""Turn the raw customer table into model inputs."""
import numpy as np

INCOME_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def build_xy(df, target, id_column):
    """Return (X, y). Drops the ID, encodes income and adds ratio features."""
    X = df.drop(columns=[c for c in (target, id_column) if c in df.columns]).copy()
    X["Income_Bracket"] = X["Income_Bracket"].map(INCOME_ORDER)
    limit = X["Credit_Limit"].replace(0, np.nan)
    X["utilization"] = X["Current_Balance"] / limit
    X["spend_to_limit"] = X["Monthly_Spend"] / limit
    X["spend_per_txn"] = X["Monthly_Spend"] / X["Num_Transactions_Monthly"].replace(0, np.nan)
    y = df[target] if target in df.columns else None
    return X, y
