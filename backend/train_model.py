"""
train_model.py
==============
PHASE 5 - Train and compare several regression models

WHAT THIS SCRIPT DOES
---------------------
1. Loads the data and splits it into a TRAINING set and a TEST set.
      - Training set  -> the model learns from this.
      - Test set      -> kept hidden during training, used only to fairly judge
                         how well the model does on data it has never seen.
2. Builds FOUR different regression models, each wrapped in a Pipeline that
   FIRST runs our Phase 4 preprocessing, THEN the model. So each pipeline can
   take raw house data and go straight to a predicted price.
3. Trains all four and measures their accuracy on the test set using 4 metrics.
4. Prints a comparison table and automatically picks the best model.

THE FOUR MODELS (in plain terms)
--------------------------------
- Linear Regression      : draws the best straight-line relationship.
- Decision Tree          : asks yes/no questions to split houses into groups.
- Random Forest          : many decision trees voting together (usually stronger).
- Gradient Boosting      : trees built one after another, each fixing the last
                           one's mistakes.

THE FOUR METRICS (lower is better, EXCEPT R2 where higher is better)
--------------------------------------------------------------------
- MAE  (Mean Absolute Error) : on average, how many rupees off are we?
- MSE  (Mean Squared Error)  : like MAE but punishes big misses much more.
- RMSE (Root MSE)            : MSE brought back to rupee units (easier to read).
- R2   (R-squared)           : fraction of price variation explained, 0 to 1.
                               1.0 = perfect, 0 = no better than guessing the average.

How to run (project root, with .venv active):
    python backend\\train_model.py
"""

import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reuse the SAME preprocessing and data-loading we defined in Phase 4.
# (Because train_model.py sits in the 'backend' folder next to preprocessing.py,
#  Python can import it directly.)
from preprocessing import build_preprocessor, load_data


# A fixed random seed. Using the same number every run means the train/test
# split and the random models come out identical each time -> reproducible.
RANDOM_STATE = 42

# Where the final trained pipeline will be saved (Phase 6).
# Path(__file__).parent is the 'backend' folder, so this points to
# backend/model/house_price_model.pkl regardless of where you run from.
MODEL_PATH = Path(__file__).parent / "model" / "house_price_model.pkl"


def main():
    print("=" * 74)
    print("PHASE 5  -  Training and comparing regression models")
    print("=" * 74)

    # -----------------------------------------------------------------------
    # 1. LOAD DATA AND SPLIT INTO TRAIN / TEST
    # -----------------------------------------------------------------------
    X, y = load_data()

    # 80% for training, 20% held back for testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"\nTotal rows: {len(X)}")
    print(f"  Training rows: {len(X_train)}")
    print(f"  Test rows    : {len(X_test)}")

    # -----------------------------------------------------------------------
    # 2. DEFINE THE FOUR MODELS
    # -----------------------------------------------------------------------
    # random_state is passed to the models that involve randomness so results
    # are reproducible.
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    # -----------------------------------------------------------------------
    # 3. TRAIN EACH MODEL AND COLLECT ITS SCORES
    # -----------------------------------------------------------------------
    results = []  # we'll store one dict of scores per model

    for name, model in models.items():
        # Build a pipeline: preprocessing FIRST, then the model.
        # A fresh preprocessor is created for each pipeline so they stay separate.
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])

        # Train (fit) on the training data only.
        pipeline.fit(X_train, y_train)

        # Predict prices for the unseen test houses.
        predictions = pipeline.predict(X_test)

        # Measure accuracy by comparing predictions to the TRUE test prices.
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)   # works on every scikit-learn version
        r2 = r2_score(y_test, predictions)

        results.append({
            "name": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        })

    # -----------------------------------------------------------------------
    # 4. PRINT A COMPARISON TABLE
    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("MODEL COMPARISON  (test set)")
    print("=" * 74)
    header = f"{'Model':<20} {'MAE':>14} {'RMSE':>14} {'R2':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<20} {r['MAE']:>14,.0f} {r['RMSE']:>14,.0f} {r['R2']:>8.4f}")

    print("\n(MAE and RMSE are in rupees - lower is better. "
          "R2 is 0 to 1 - higher is better.)")

    # -----------------------------------------------------------------------
    # 5. PICK THE BEST MODEL (highest R2 on the test set)
    # -----------------------------------------------------------------------
    best = max(results, key=lambda r: r["R2"])
    print("\n" + "=" * 74)
    print(f"BEST MODEL: {best['name']}")
    print(f"  It explains about {best['R2'] * 100:.1f}% of the variation in price")
    print(f"  and is on average about Rs {best['MAE']:,.0f} off per house.")
    print("=" * 74)

    # -----------------------------------------------------------------------
    # 6. SAVE THE WINNING PIPELINE  (PHASE 6)
    # -----------------------------------------------------------------------
    # We now build one final pipeline using the WINNING model, and train it on
    # the FULL dataset (all 545 rows, not just the 80% training slice).
    #
    # Why refit on everything? The train/test split was only needed to fairly
    # COMPARE models. Now that Gradient Boosting has won, we want the strongest
    # possible final model, so we let it learn from every row we have.
    #
    # The saved pipeline contains BOTH the preprocessing AND the model. That is
    # the whole point: the API can hand it a raw house (with "yes"/"no" text and
    # all) and the pipeline does preprocessing + prediction in one step. There is
    # no need to duplicate any preprocessing code inside the API.
    print("\nTraining the winning model on the FULL dataset and saving it...")

    best_model = models[best["name"]]  # reuse the winning model object
    final_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", best_model),
    ])
    # Calling .fit() again trains it from scratch on ALL the data.
    final_pipeline.fit(X, y)

    # Make sure the 'model' folder exists, then save the pipeline to a .pkl file.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)

    print(f"Saved trained pipeline to: {MODEL_PATH}")
    print("\nThis single .pkl file = preprocessing + model together.")
    print("Phase 7 will load it and make a real prediction.")


if __name__ == "__main__":
    main()
