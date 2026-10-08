"""
data_loader.py

Shared loading, cleaning, feature and scoring code for the accident severity models.
Phase 2 builds the statistical baselines with these pieces, and later phases reuse them
so every model is compared on exactly the same data and the same scoring.

Typical use:

    from data_loader import load_accident_table, setup_masks, grouped_folds, fit_predict_all, score

    table = load_accident_table()
    masks = setup_masks(table)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TABLE_NAME = "accidents_labeled_all_industries.csv"

LABELS = [1, 2, 3]
LABEL_NAMES = {1: "Fatality", 2: "Hospitalized", 3: "Not hospitalized"}

CAT_COLS = ["event_type", "src_of_injury", "evn_factor", "hum_factor", "task_assigned", "occ_code",
            "const_op", "const_op_cause", "const_end_use", "project_cost", "project_type"]
NUM_COLS = ["age", "build_stories", "nonbuild_ht"]
STRUCT_COLS = CAT_COLS + NUM_COLS

MODEL_ORDER = ["Dummy most frequent", "Dummy stratified", "Rule based (event type)",
               "Logistic Regression", "Ridge Classifier", "Naive Bayes"]

OUTCOME_STEMS = (
    r"\b(?:injur\w*|fractur\w*|amputat\w*|paraly\w*|lacerat\w*|contus\w*|sprain\w*|bruis\w*|stitch\w*|"
    r"concuss\w*|dislocat\w*|treat\w*|releas\w*|discharg\w*|requir\w*|underw\w*|undergo\w*|hospitali\w*|"
    r"medical\w*|minor|uninjur\w*|later|scene|arriv\w*|examiner|coroner|autops\w*|complication\w*|"
    r"cardiac|arrest|cpr|resuscitat\w*|unresponsive|lifeless|pulse|massive|internal|blunt|days|weeks|"
    r"months|recover\w*|surviv\w*|condition|critical\w*|serious\w*|severe\w*|sustain\w*|suffer\w*|"
    r"incur\w*|receiv\w*)\b"
)
GLUED_OUTCOME = r"(?:hospitali\w*|killed|admitted|treated|died)"
YEARS = r"\b(?:19|20)\d{2}\b"


def find_project_root():
    """Return the project root whether code runs from the root, notebooks/ or src/."""
    here = Path.cwd()
    if here.name in ("notebooks", "src"):
        return here.parent
    return here


def clean_event_text(text):
    """
    Remove outcome language and years so the text describes the event, not its result.

    Order matters. Outcome words glued to other words (for example "admittedlater") are removed
    first, because taking them out can expose a year or an outcome word that was hidden next to them.
    """
    return (text.str.replace(GLUED_OUTCOME, " ", regex=True)
                .str.replace(YEARS, " ", regex=True)
                .str.replace(OUTCOME_STEMS, " ", regex=True))


def load_accident_table(path=None, sample_accidents=None, random_state=RANDOM_STATE):
    """
    Load the prepared accident table and apply the final text cleaning.

    sample_accidents: if given, keep only that many randomly chosen accidents (for quick tests).
    The original report text is kept in the text_report column for comparison.
    """
    if path is None:
        path = find_project_root() / "data" / "processed" / TABLE_NAME
    table = pd.read_csv(path, low_memory=False)
    table["event_date"] = pd.to_datetime(table["event_date"], errors="coerce", utc=True)
    table["text_clean"] = table["text_clean"].fillna("")

    if sample_accidents:
        rng = np.random.default_rng(random_state)
        ids = table["summary_nr"].unique()
        keep = rng.choice(ids, size=min(sample_accidents, len(ids)), replace=False)
        table = table[table["summary_nr"].isin(keep)]
    table = table.reset_index(drop=True)

    table["text_report"] = table["text_clean"]
    table["text_clean"] = clean_event_text(table["text_report"])
    for c in CAT_COLS:
        table[c] = table[c].astype(str)
    return table


def setup_masks(table):
    """Training sets compared in Phase 2, plus the mask of cases every model is scored on."""
    return {
        "Construction only (strict)": table["is_construction_strict"].to_numpy(),
        "All industries": np.ones(len(table), dtype=bool),
        "Construction broad": table["is_construction_broad"].to_numpy(),
    }


def test_mask(table):
    """Models are always scored on strict construction cases."""
    return table["is_construction_strict"].to_numpy()


def make_struct():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, dtype=np.float32), CAT_COLS),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUM_COLS),
    ])


def make_tfidf():
    return TfidfVectorizer(max_features=20000, min_df=5, stop_words="english",
                           sublinear_tf=True, dtype=np.float32)


def grouped_folds(table, n_splits=5, seed=RANDOM_STATE):
    """Folds that keep all rows of one accident together and balance the severity classes."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return sgkf.split(table, table["severity_3class"], groups=table["summary_nr"])


def score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    yt, yp = (y_true == 1), (y_pred == 1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "f1_fatal": per_class[0],
        "f1_hospitalized": per_class[1],
        "f1_not_hospitalized": per_class[2],
        "binary_accuracy": accuracy_score(yt, yp),
        "binary_macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
    }


def fit_predict_all(train_df, test_df, seed=RANDOM_STATE):
    """Fit every baseline on train_df and return a dict of predictions for test_df."""
    y_tr = train_df["severity_3class"].to_numpy()
    classes, counts = np.unique(y_tr, return_counts=True)
    majority = classes[counts.argmax()]
    preds = {}

    preds["Dummy most frequent"] = np.full(len(test_df), majority)
    rng = np.random.default_rng(seed)
    preds["Dummy stratified"] = rng.choice(classes, size=len(test_df), p=counts / counts.sum())

    rule = train_df.groupby("event_type")["severity_3class"].agg(lambda s: s.mode().iloc[0])
    preds["Rule based (event type)"] = test_df["event_type"].map(rule).fillna(majority).astype(int).to_numpy()

    struct = make_struct()
    Xs_tr = struct.fit_transform(train_df[STRUCT_COLS])
    Xs_te = struct.transform(test_df[STRUCT_COLS])
    tfidf = make_tfidf()
    T_tr = tfidf.fit_transform(train_df["text_clean"])
    T_te = tfidf.transform(test_df["text_clean"])
    X_tr = sparse.hstack([Xs_tr, T_tr]).tocsr()
    X_te = sparse.hstack([Xs_te, T_te]).tocsr()

    lr = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced")
    preds["Logistic Regression"] = lr.fit(X_tr, y_tr).predict(X_te)

    ridge = RidgeClassifier(alpha=1.0, class_weight="balanced")
    preds["Ridge Classifier"] = ridge.fit(X_tr, y_tr).predict(X_te)

    nb = MultinomialNB(alpha=0.1, fit_prior=False)
    preds["Naive Bayes"] = nb.fit(T_tr, y_tr).predict(T_te)
    return preds


def macro_f1_from_cm(cm):
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean()


def group_bootstrap_ci(y_true, y_pred, groups, n_boot=500, seed=RANDOM_STATE):
    """95 percent interval for macro F1, resampling whole accidents instead of single rows."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    codes = (y_true - 1) * 3 + (y_pred - 1)
    g_codes, g_index = pd.factorize(groups)
    contrib = np.zeros((len(g_index), 9))
    np.add.at(contrib, (g_codes, codes), 1)
    rng = np.random.default_rng(seed)
    n = len(g_index)
    vals = [macro_f1_from_cm((rng.multinomial(n, np.full(n, 1.0 / n)) @ contrib).reshape(3, 3))
            for _ in range(n_boot)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def log_model_result(phase, model_name, accuracy, macro_f1, notes=""):
    """Insert one row into the model_results table in PostgreSQL (used by later phases too)."""
    import sys
    sys.path.append(str(find_project_root() / "src"))
    from db import get_db

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO model_results (phase, model_name, accuracy, macro_f1, notes) "
                "VALUES (%s, %s, %s, %s, %s)",
                (phase, model_name, float(accuracy), float(macro_f1), notes),
            )
