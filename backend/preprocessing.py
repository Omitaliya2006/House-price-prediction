"""
preprocessing.py
================
PHASE 4 - Data Preprocessing Pipeline

WHY THIS FILE EXISTS
--------------------
A machine learning model can only do maths on NUMBERS. But our dataset contains
text like "yes", "no", and "furnished". So before training we must TRANSFORM the
raw data into an all-numeric form. That transformation is called "preprocessing".

THE GOLDEN RULE
---------------
The EXACT SAME preprocessing must be applied both when we TRAIN the model and
later when a user asks for a PREDICTION. If they differ even slightly, the model
sees data in a shape it doesn't recognise and gives nonsense answers.

To guarantee they match, we define the preprocessing ONCE, here, in a function
called build_preprocessor(). Phase 5 (training) will import this function, and we
will bundle it together with the model into a single saved file. That way the
preprocessing literally travels with the model.

WHAT WE DO TO EACH COLUMN
-------------------------
1. Numerical features (area, bedrooms, bathrooms, stories, parking):
   -> StandardScaler: rescales each number so they are on a comparable scale
      (mean 0, standard deviation 1). This helps models like Linear Regression,
      and does no harm to the tree-based models.

2. Categorical features (all the yes/no columns + furnishingstatus):
   -> OneHotEncoder: turns each text category into its own 0/1 column.
      Example: furnishingstatus becomes THREE columns:
         furnishingstatus_furnished, furnishingstatus_semi-furnished,
         furnishingstatus_unfurnished
      A "furnished" house gets a 1 in the first and 0 in the others.

We wire these two transformers together with a ColumnTransformer, which applies
the right transformer to the right columns automatically.

You can RUN this file directly to SEE the transformation happen:
    python backend\\preprocessing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ---------------------------------------------------------------------------
# 1. DEFINE WHICH COLUMNS ARE WHICH
# ---------------------------------------------------------------------------
# These lists are the single source of truth for the whole project.
# (They match exactly what the Phase 3 EDA script discovered.)

# The column we want to PREDICT.
TARGET = "price"

# Number columns (get scaled).
NUMERICAL_FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
    "total_rooms",
    "amenity_count",
    "area_per_bedroom",
    "log_area",
]

# Text columns (get one-hot encoded). This includes both the yes/no columns
# and the multi-class furnishingstatus column - OneHotEncoder handles both.
CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

# The full list of input features, in a fixed order. Keeping this order stable
# matters because the API will build its input rows in this same order.
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add stable, domain-informed features while preserving raw inputs."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = X.copy()
        features["total_rooms"] = features["bedrooms"] + features["bathrooms"]
        features["amenity_count"] = features[
            [
                "mainroad",
                "guestroom",
                "basement",
                "hotwaterheating",
                "airconditioning",
                "prefarea",
            ]
        ].eq("yes").sum(axis=1)
        features["area_per_bedroom"] = features["area"] / features["bedrooms"].clip(lower=1)
        features["log_area"] = np.log1p(features["area"])
        return features


# ---------------------------------------------------------------------------
# 2. BUILD THE PREPROCESSOR
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """
    Create and return the preprocessing transformer.

    We return a *fresh, unfitted* ColumnTransformer every time. It only learns
    from data later, when .fit() is called on it during training. Returning it
    unfitted keeps this function reusable and side-effect free.
    """
    # StandardScaler for the numeric columns.
    numeric_transformer = StandardScaler()

    # OneHotEncoder for the text columns.
    #   handle_unknown="ignore" -> if a category we never saw in training shows
    #       up at prediction time, don't crash; just encode it as all zeros.
    #   sparse_output=False -> return a normal (dense) array that is easy to read.
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    # ColumnTransformer routes each group of columns to its transformer.
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[
        ("feature_engineering", HousingFeatureEngineer()),
        ("columns", column_transformer),
    ])


# ---------------------------------------------------------------------------
# 3. HELPER: LOAD THE DATASET SPLIT INTO FEATURES (X) AND TARGET (y)
# ---------------------------------------------------------------------------
def load_data():
    """
    Load Housing.csv and split it into:
        X -> the input features (everything except price)
        y -> the target we want to predict (price)
    """
    csv_path = Path(__file__).parent / "data" / "Housing.csv"
    df = pd.read_csv(csv_path)
    X = df[CATEGORICAL_FEATURES + [
        "area", "bedrooms", "bathrooms", "stories", "parking"
    ]]
    y = df[TARGET]         # just the price column
    return X, y


# ---------------------------------------------------------------------------
# 4. DEMO: run this file directly to SEE preprocessing in action
# ---------------------------------------------------------------------------
# The code under this "if" only runs when you execute:  python backend\preprocessing.py
# It does NOT run when another file merely imports build_preprocessor().
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4  -  Preprocessing demonstration")
    print("=" * 70)

    # Load the data.
    X, y = load_data()
    print(f"\nRaw feature table shape (rows, columns): {X.shape}")
    print("\nFirst 3 rows of RAW features (note the text values):")
    print(X.head(3).to_string(index=False))

    # Build the preprocessor and fit+transform the data so we can inspect it.
    # fit_transform = 'learn from this data' + 'apply the transformation'.
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # After one-hot encoding, we have MORE columns than we started with.
    feature_names = preprocessor.get_feature_names_out()
    print(f"\nTransformed array shape (rows, columns): {X_transformed.shape}")
    print(f"Number of columns went from {X.shape[1]} (raw) "
          f"to {X_transformed.shape[1]} (after encoding).")

    print(f"\nThe {len(feature_names)} columns produced by preprocessing:")
    for i, name in enumerate(feature_names, start=1):
        print(f"  {i:>2}. {name}")

    print("\nFirst row AFTER preprocessing (all numbers now):")
    # Round for readability so the scaled decimals aren't too long.
    print("  ", [round(float(v), 3) for v in X_transformed[0]])

    print("\n" + "=" * 70)
    print("Preprocessing works. In Phase 5 we attach a model AFTER this step,")
    print("so raw house data flows: raw -> preprocessing -> model -> price.")
    print("=" * 70)
