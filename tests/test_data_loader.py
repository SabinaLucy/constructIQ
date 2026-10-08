import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_loader as dl

TABLE_PATH = ROOT / "data" / "processed" / dl.TABLE_NAME
SAMPLE_ACCIDENTS = 8000


@pytest.fixture(scope="module")
def table():
    if not TABLE_PATH.exists():
        pytest.skip(f"Prepared table not found at {TABLE_PATH}")
    return dl.load_accident_table(path=TABLE_PATH, sample_accidents=SAMPLE_ACCIDENTS)


def test_table_has_required_columns(table):
    needed = ["summary_nr", "event_date", "severity_3class", "is_fatal", "is_construction_strict",
              "is_construction_broad", "text_clean", "text_report"] + dl.STRUCT_COLS
    assert [c for c in needed if c not in table.columns] == []
    assert set(table["severity_3class"].unique()) <= {1, 2, 3}


def test_outcome_language_is_removed(table):
    for name, pattern in [("outcome words", dl.OUTCOME_STEMS), ("years", dl.YEARS)]:
        bad = table.loc[table["text_clean"].str.contains(pattern, regex=True), "text_clean"]
        assert bad.empty, f"{len(bad)} texts still contain {name}, for example: {bad.iloc[0][:200]}"


def test_cleaning_order_handles_glued_words():
    import pandas as pd
    s = pd.Series(["he was admittedlater that day", "the worker diedscene was secured", "killed2015 on site"])
    cleaned = dl.clean_event_text(s)
    assert not cleaned.str.contains(dl.OUTCOME_STEMS, regex=True).any()
    assert not cleaned.str.contains(dl.YEARS, regex=True).any()


def test_setup_masks_are_consistent(table):
    masks = dl.setup_masks(table)
    strict = masks["Construction only (strict)"]
    broad = masks["Construction broad"]
    assert strict.sum() > 0
    assert (broad | ~strict).all()          # strict is a subset of broad
    assert masks["All industries"].all()
    assert (dl.test_mask(table) == strict).all()


def test_folds_keep_each_accident_together(table):
    for train_idx, test_idx in dl.grouped_folds(table, n_splits=3):
        train_ids = set(table.iloc[train_idx]["summary_nr"])
        test_ids = set(table.iloc[test_idx]["summary_nr"])
        assert train_ids.isdisjoint(test_ids)


def test_score_perfect_and_wrong():
    y = np.array([1, 2, 3, 1, 2, 3])
    assert dl.score(y, y)["macro_f1"] == pytest.approx(1.0)
    assert dl.score(y, np.full(6, 1))["macro_f1"] < 0.5


def test_baselines_run_and_beat_dummy(table):
    masks = dl.setup_masks(table)
    train_idx, test_idx = next(dl.grouped_folds(table, n_splits=3))
    test_pos = test_idx[dl.test_mask(table)[test_idx]]
    train_pos = train_idx[masks["All industries"][train_idx]]
    train_df, test_df = table.iloc[train_pos], table.iloc[test_pos]

    preds = dl.fit_predict_all(train_df, test_df)
    assert list(preds) == dl.MODEL_ORDER
    f1 = {}
    for name, p in preds.items():
        assert len(p) == len(test_df)
        assert set(np.unique(p)) <= {1, 2, 3}
        f1[name] = dl.score(test_df["severity_3class"], p)["macro_f1"]
    assert f1["Logistic Regression"] > f1["Dummy most frequent"] + 0.1


def test_bootstrap_interval_is_ordered():
    rng = np.random.default_rng(0)
    y = rng.integers(1, 4, 600)
    p = np.where(rng.random(600) < 0.7, y, rng.integers(1, 4, 600))
    groups = rng.integers(0, 400, 600)
    low, high = dl.group_bootstrap_ci(y, p, groups, n_boot=100)
    assert 0 <= low <= high <= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
