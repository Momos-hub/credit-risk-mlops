import numpy as np
import pandas as pd
import yaml

from src.features import build_xy

CFG = yaml.safe_load(open("config.yaml"))


def load():
    return pd.read_csv(CFG["data"]["path"])


def test_id_and_target_removed():
    X, y = build_xy(load(), CFG["data"]["target"], CFG["data"]["id_column"])
    assert "Customer_ID" not in X.columns
    assert "High_Risk_Customer" not in X.columns
    assert len(X) == len(y)


def test_income_is_numeric():
    X, _ = build_xy(load(), CFG["data"]["target"], CFG["data"]["id_column"])
    assert set(X["Income_Bracket"].unique()) <= {0, 1, 2}


def test_utilization_is_balance_over_limit():
    df = load().dropna()
    X, _ = build_xy(df, CFG["data"]["target"], CFG["data"]["id_column"])
    assert np.allclose(X["utilization"], df["Current_Balance"] / df["Credit_Limit"])
