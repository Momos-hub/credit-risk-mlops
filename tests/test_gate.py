from src.quality_gate import check

GATE = {"min_auc": 0.90, "min_recall": 0.75, "min_precision": 0.35, "min_ks": 0.60,
        "max_auc_drop_vs_baseline": 0.03}
GOOD = {"auc": 0.95, "recall": 0.85, "precision": 0.50, "ks": 0.75}


def test_good_model_passes():
    assert check(GOOD, GATE) == []


def test_low_auc_fails():
    assert check({**GOOD, "auc": 0.80}, GATE)


def test_low_recall_fails():
    assert check({**GOOD, "recall": 0.5}, GATE)


def test_flag_everyone_model_fails_on_precision():
    assert check({**GOOD, "precision": 0.19}, GATE)


def test_regression_vs_baseline_fails():
    assert check({**GOOD, "auc": 0.91}, GATE, {"auc": 0.95})


def test_small_drop_vs_baseline_is_ok():
    assert check({**GOOD, "auc": 0.94}, GATE, {"auc": 0.95}) == []
