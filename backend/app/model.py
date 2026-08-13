"""
model.py
========
PHASE 8 - Loading the trained model and turning house features into a price

This file is the bridge between the saved .pkl file and the API. It keeps ALL
machine-learning concerns in one place, so main.py can focus purely on the web
side. (This follows the project rule: keep ML code separate from API code.)

Key idea: we load the model ONCE when the server starts (loading is a little
slow), then reuse that in-memory model for every request (which is fast).
"""

import joblib
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# LOCATE THE SAVED MODEL
# ---------------------------------------------------------------------------
# This file is at:  backend/app/model.py
# .parent          -> backend/app
# .parent.parent   -> backend
# So the model path becomes: backend/model/house_price_model.pkl
MODEL_PATH = Path(__file__).parent.parent / "model" / "house_price_model.pkl"


class ModelService:
    """A small helper that loads the pipeline and makes predictions."""

    def __init__(self):
        # We start with no model loaded. It gets loaded by load() below.
        self.model = None

    def load(self):
        """
        Load the saved pipeline into memory. Called once at server startup.
        Raises a clear error if the file is missing, so problems are obvious.
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                f"Run 'python backend\\train_model.py' first to create it."
            )
        self.model = joblib.load(MODEL_PATH)
        return self.model

    def is_ready(self) -> bool:
        """True once the model has been loaded successfully."""
        return self.model is not None

    def predict(self, features: dict) -> float:
        """
        Take a dict of raw house features and return the predicted price.

        We build a 1-row pandas DataFrame from the features because the pipeline
        expects a table with named columns (just like in Phase 7). The pipeline
        does preprocessing + prediction internally.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        input_df = pd.DataFrame([features])
        prediction = self.model.predict(input_df)[0]
        return float(prediction)


# A single shared instance the rest of the app imports and uses.
model_service = ModelService()
