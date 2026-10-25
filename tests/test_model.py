import pandas as pd
import yaml

from src.train import train_model

CFG = yaml.safe_load(open("config.yaml"))


def test_model_trains_and_beats_chance():
    df = pd.read_csv(CFG["data"]["path"])
    _, metrics, preds, _ = train_model(df, CFG)
    assert metrics["auc"] > 0.8
    assert set(preds["pred"].unique()) <= {0, 1}
    assert len(preds) == int(len(df) * CFG["model"]["test_size"])


def test_training_is_reproducible():
    df = pd.read_csv(CFG["data"]["path"])
    _, m1, _, _ = train_model(df, CFG)
    _, m2, _, _ = train_model(df, CFG)
    assert m1 == m2
