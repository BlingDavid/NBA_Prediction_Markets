import pandas as pd
from live_bootstrap_model import filter_first_half


def test_keeps_only_period_1_and_2():
    df = pd.DataFrame({"period": [0, 1, 2, 3, 4], "x": range(5)})
    out = filter_first_half(df)
    assert sorted(out["period"].tolist()) == [1, 2]


def test_noop_when_flag_false():
    df = pd.DataFrame({"period": [0, 1, 2, 3, 4]})
    assert len(filter_first_half(df, enabled=False)) == 5


def test_noop_when_no_period_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert len(filter_first_half(df)) == 3
