"""
data_analysis.py
================
PHASE 3 - Exploratory Data Analysis (EDA)

Purpose:
    Before we train any machine learning model, we must first UNDERSTAND our data.
    This script loads Housing.csv and prints a series of reports that answer
    basic but important questions:
        - What does the data look like?
        - How big is it?
        - What type is each column?
        - Is any data missing or duplicated?
        - What are the typical/average values?
        - Which columns are numbers vs. categories (text)?

    This script ONLY reads and describes the data. It does not change anything.
    Run it as many times as you like - it is completely safe.

How to run (from the project ROOT, with the .venv active):
    python backend\\data_analysis.py
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
# 'pandas' is the library for working with tables of data (like a spreadsheet).
# We give it the nickname 'pd' by convention so we can type less.
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# 2. LOCATE AND LOAD THE DATASET
# ---------------------------------------------------------------------------
# We build the path to Housing.csv relative to THIS script's location.
# Path(__file__) is this file (data_analysis.py). .parent is the 'backend' folder.
# So the CSV path becomes:  backend/data/Housing.csv
# Doing it this way means the script works no matter which folder you run it from.
CSV_PATH = Path(__file__).parent / "data" / "Housing.csv"

print("=" * 70)
print("HOUSE PRICE PREDICTOR  -  Exploratory Data Analysis (EDA)")
print("=" * 70)
print(f"Loading dataset from: {CSV_PATH}")

# read_csv reads the CSV file into a 'DataFrame' - pandas' name for a table.
df = pd.read_csv(CSV_PATH)
print("Dataset loaded successfully.\n")


# ---------------------------------------------------------------------------
# 3. FIRST 5 ROWS
# ---------------------------------------------------------------------------
# .head() shows the top rows so we can eyeball what the data actually looks like.
print("-" * 70)
print("1) FIRST 5 ROWS  (df.head())")
print("-" * 70)
print(df.head())
print()


# ---------------------------------------------------------------------------
# 4. SHAPE (rows x columns)
# ---------------------------------------------------------------------------
# .shape gives (number_of_rows, number_of_columns).
print("-" * 70)
print("2) SHAPE  (rows, columns)")
print("-" * 70)
rows, cols = df.shape
print(f"The dataset has {rows} rows and {cols} columns.\n")


# ---------------------------------------------------------------------------
# 5. COLUMN NAMES
# ---------------------------------------------------------------------------
print("-" * 70)
print("3) COLUMN NAMES")
print("-" * 70)
for i, name in enumerate(df.columns, start=1):
    print(f"  {i:>2}. {name}")
print()


# ---------------------------------------------------------------------------
# 6. DATA TYPES
# ---------------------------------------------------------------------------
# .dtypes tells us how pandas is storing each column:
#   int64   = whole numbers      (e.g. bedrooms)
#   float64 = decimal numbers
#   object  = text / categories  (e.g. "yes"/"no", "furnished")
print("-" * 70)
print("4) DATA TYPES  (df.dtypes)")
print("-" * 70)
print(df.dtypes)
print()


# ---------------------------------------------------------------------------
# 7. MISSING VALUES
# ---------------------------------------------------------------------------
# .isnull() marks each empty cell as True; .sum() counts them per column.
# We WANT all zeros - that means no data is missing.
print("-" * 70)
print("5) MISSING VALUES per column")
print("-" * 70)
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing values in the whole dataset: {int(missing.sum())}\n")


# ---------------------------------------------------------------------------
# 8. DUPLICATE ROWS
# ---------------------------------------------------------------------------
# .duplicated() flags rows that are exact copies of an earlier row.
print("-" * 70)
print("6) DUPLICATE ROWS")
print("-" * 70)
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {int(duplicate_count)}\n")


# ---------------------------------------------------------------------------
# 9. STATISTICAL SUMMARY (numeric columns)
# ---------------------------------------------------------------------------
# .describe() gives count, mean (average), std (spread), min, quartiles, max
# for every NUMERIC column. Great for spotting the scale/range of each feature.
print("-" * 70)
print("7) STATISTICAL SUMMARY  (df.describe())")
print("-" * 70)
print(df.describe())
print()


# ---------------------------------------------------------------------------
# 10. UNIQUE VALUES FOR CATEGORICAL (TEXT) COLUMNS
# ---------------------------------------------------------------------------
# For text columns we list the distinct values and how often each appears.
# This confirms, e.g., that 'mainroad' only ever contains "yes" or "no".
print("-" * 70)
print("8) UNIQUE VALUES for each categorical (text) column")
print("-" * 70)
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    print(f"\n  Column '{col}':")
    print(f"    Unique values -> {list(df[col].unique())}")
    # value_counts() shows how many rows fall into each category.
    counts = df[col].value_counts()
    for value, count in counts.items():
        print(f"      {value:<15} : {count}")
print()


# ---------------------------------------------------------------------------
# 11. IDENTIFY NUMERICAL vs CATEGORICAL FEATURES
# ---------------------------------------------------------------------------
# We separate the columns into two groups. This is exactly the split we will
# use in Phase 4 when we build the preprocessing pipeline.
# NOTE: 'price' is our TARGET (what we predict), so we exclude it from features.
print("-" * 70)
print("9) FEATURE TYPES  (used later for preprocessing)")
print("-" * 70)

numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = df.select_dtypes(include="object").columns.tolist()

# Remove the target 'price' from the numerical list - it is not an input feature.
if "price" in numerical_features:
    numerical_features.remove("price")

print(f"  TARGET (what we predict): price")
print(f"\n  Numerical features ({len(numerical_features)}):")
print(f"    {numerical_features}")
print(f"\n  Categorical features ({len(categorical_features)}):")
print(f"    {categorical_features}")
print()

print("=" * 70)
print("EDA complete. The data is clean and ready for preprocessing (Phase 4).")
print("=" * 70)
