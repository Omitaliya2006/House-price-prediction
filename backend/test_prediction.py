"""
test_prediction.py
===================
PHASE 7 - Load the saved model and make a real prediction

WHY THIS SCRIPT MATTERS
-----------------------
In Phase 6 we saved our trained pipeline to a file (house_price_model.pkl).
This script proves the whole thing actually works OUTSIDE of training:
    1. It loads that saved file (no retraining - instant).
    2. It feeds in ONE sample house described in plain, raw form
       (with "yes"/"no" text, exactly like a user would enter on a website).
    3. The pipeline does preprocessing + prediction internally and gives a price.

This is a mini rehearsal for what the API will do in Phase 8. The API will
receive a house as JSON, but the idea is identical: raw house in -> price out.

How to run (project root, with .venv active):
    python backend\\test_prediction.py
"""

import joblib
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. LOCATE AND LOAD THE SAVED MODEL
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model" / "house_price_model.pkl"

print("=" * 70)
print("PHASE 7  -  Test prediction using the saved model")
print("=" * 70)
print(f"Loading model from: {MODEL_PATH}")

# joblib.load rebuilds the exact pipeline we saved (preprocessing + model).
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.\n")


# ---------------------------------------------------------------------------
# 2. DESCRIBE ONE SAMPLE HOUSE
# ---------------------------------------------------------------------------
# We write the house as a normal Python dictionary using RAW values - the same
# "yes"/"no"/"furnished" wording as the original CSV. We do NOT do any manual
# encoding or scaling here; the pipeline handles all of that for us.
#
# IMPORTANT: the KEYS must exactly match the dataset column names, and we must
# include every feature the model was trained on.
sample_house = {
    "area": 5000,
    "bedrooms": 4,
    "bathrooms": 2,
    "stories": 2,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "yes",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 2,
    "prefarea": "yes",
    "furnishingstatus": "furnished",
}

print("Sample house details:")
for key, value in sample_house.items():
    print(f"  {key:<18}: {value}")


# ---------------------------------------------------------------------------
# 3. CONVERT THE HOUSE INTO THE SHAPE THE MODEL EXPECTS
# ---------------------------------------------------------------------------
# scikit-learn pipelines expect a table (DataFrame), not a single dictionary.
# We wrap our one house in a list so pandas builds a 1-row table.
# The column names in this table let the ColumnTransformer route each value to
# the correct transformer (numeric vs. categorical).
input_df = pd.DataFrame([sample_house])


# ---------------------------------------------------------------------------
# 4. PREDICT
# ---------------------------------------------------------------------------
# .predict() returns an array of predictions (one per input row). We have one
# house, so we take the first (and only) element.
predicted_price = model.predict(input_df)[0]

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)
# :,.0f formats the number with thousands separators and no decimals.
print(f"  Estimated house price: Rs {predicted_price:,.0f}")
print("=" * 70)

# ---------------------------------------------------------------------------
# 5. HONEST DISCLAIMER
# ---------------------------------------------------------------------------
print("\nNote: This estimate is produced by a model trained on 545 houses from")
print("the Housing.csv dataset. It reflects PATTERNS IN THAT DATA ONLY and is")
print("NOT a guaranteed or official market price. Real prices also depend on")
print("location, condition, timing and many factors not present in the data.")
