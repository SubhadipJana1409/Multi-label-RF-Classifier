"""
================================================================
predict.py — Use the trained AMR Multi-label RF Model
================================================================
How to use:
    python predict.py

Or import and call predict_isolate() in your own code.
================================================================
"""

import pickle
import numpy as np
import pandas as pd

# ── Load models ──
with open("multilabel_rf_model.pkl",  "rb") as f: model    = pickle.load(f)
with open("model_metadata.pkl",       "rb") as f: meta     = pickle.load(f)
with open("individual_ab_models.pkl", "rb") as f: ind_models = pickle.load(f)

TARGET_ABS   = meta["target_antibiotics"]
FEATURE_COLS = meta["feature_columns"]
TOP_SPECIES  = meta["top_species"]

AB_FULLNAMES = {
    "ERY":"Erythromycin","AZM":"Azithromycin","AMC":"Amoxicillin-clav",
    "VAN":"Vancomycin","GEN":"Gentamicin","CAZ":"Ceftazidime",
    "CXM":"Cefuroxime","SXT":"Trimethoprim-sulfa","PEN":"Penicillin",
    "CIP":"Ciprofloxacin","CLI":"Clindamycin","TMP":"Trimethoprim",
}


def build_feature_vector(species, ward, age, gender, year=2015):
    """
    Build feature vector for one isolate.

    Parameters
    ----------
    species : str   e.g. "B_ESCHR_COLI"
    ward    : str   "ICU" | "Clinical" | "Outpatient"
    age     : int   patient age in years
    gender  : str   "M" | "F"
    year    : int   isolation year (default 2015)

    Returns
    -------
    pd.DataFrame — one-row feature matrix aligned to training columns
    """
    sp_grp = species if species in TOP_SPECIES else "Other"

    row = {col: 0.0 for col in FEATURE_COLS}

    sp_col   = f"sp_{sp_grp}"
    ward_col = f"ward_{ward}"
    if sp_col   in row: row[sp_col]   = 1.0
    if ward_col in row: row[ward_col] = 1.0

    row["gender_M"] = 1.0 if gender == "M" else 0.0
    row["age"]      = (age  - meta["age_mean"])  / meta["age_std"]
    row["year"]     = (year - meta["year_mean"]) / meta["year_std"]

    return pd.DataFrame([row])[FEATURE_COLS]


def predict_isolate(species, ward, age, gender, year=2015, threshold=0.5):
    """
    Predict resistance profile for a single isolate.

    Returns
    -------
    dict with keys: predictions, probabilities, resistant_to
    """
    X = build_feature_vector(species, ward, age, gender, year)

    probs = {ab: ind_models[ab].predict_proba(X)[0][1]
             for ab in TARGET_ABS}
    preds = {ab: int(p >= threshold) for ab, p in probs.items()}
    resistant_to = [AB_FULLNAMES[ab] for ab, r in preds.items() if r == 1]

    return {
        "predictions"  : preds,
        "probabilities": {ab: round(p, 3) for ab, p in probs.items()},
        "resistant_to" : resistant_to,
        "n_resistant"  : len(resistant_to),
    }


# ── Demo ──
if __name__ == "__main__":
    print("=" * 55)
    print("AMR Resistance Predictor — Day 10 Model")
    print("=" * 55)

    test_cases = [
        ("B_ESCHR_COLI",  "ICU",        75, "M", 2015),
        ("B_STPHY_AURS",  "Clinical",   45, "F", 2012),
        ("B_STPHY_AURS",  "ICU",        82, "M", 2017),
        ("B_KLBSL_PNMN",  "ICU",        60, "F", 2016),
        ("B_STRPT_PNMN",  "Outpatient", 30, "M", 2010),
    ]

    for species, ward, age, gender, year in test_cases:
        result = predict_isolate(species, ward, age, gender, year)
        print(f"\nIsolate: {species} | {ward} | Age={age} | {gender} | {year}")
        print(f"  Resistant to ({result['n_resistant']} antibiotics):")
        for ab in TARGET_ABS:
            prob = result["probabilities"][ab]
            pred = result["predictions"][ab]
            flag = "🔴 R" if pred else "🟢 S"
            print(f"    {ab:5s} {flag}  (p={prob:.3f})")
