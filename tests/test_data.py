import pandas as pd
import pytest
import yaml

from src.data_validation import validate

CFG = yaml.safe_load(open("config.yaml"))


@pytest.fixture
def df():
    return pd.read_csv(CFG["data"]["path"])


def test_real_data_is_valid(df):
    assert validate(df, CFG) == []


def test_both_versions_are_valid():
    for path in ("data/credit_card_customers_v1.csv", "data/credit_card_customers_v2.csv"):
        assert validate(pd.read_csv(path), CFG) == []


def test_negative_age_is_caught(df):
    df.loc[0, "Age"] = -5
    assert any("Age" in p for p in validate(df, CFG))


def test_too_many_nulls_is_caught(df):
    df.loc[: int(0.3 * len(df)), "Credit_Limit"] = None
    assert any("Credit_Limit" in p for p in validate(df, CFG))


def test_unknown_income_level_is_caught(df):
    df.loc[0, "Income_Bracket"] = "Platinum"
    assert any("Income_Bracket" in p for p in validate(df, CFG))


def test_missing_column_is_caught(df):
    assert validate(df.drop(columns=["Tenure_Months"]), CFG)


def test_duplicate_ids_are_caught(df):
    df.loc[1, "Customer_ID"] = df.loc[0, "Customer_ID"]
    assert any("duplicate" in p for p in validate(df, CFG))
