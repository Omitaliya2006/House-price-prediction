"""
schemas.py
==========
PHASE 8 - Request and response "shapes" (data models) for the API

WHAT IS A SCHEMA?
-----------------
When the website sends house details to our API, we must make sure the data is
COMPLETE and VALID before feeding it to the model. Pydantic lets us describe the
exact shape we expect. If a request is missing a field, has the wrong type, or an
invalid value (e.g. mainroad = "maybe"), FastAPI automatically rejects it with a
clear error - we don't have to write that checking code ourselves.

There are two schemas here:
  - HouseFeatures     : the INPUT the API expects (one house).
  - PredictionResponse: the OUTPUT the API sends back (the predicted price).
"""

from typing import Literal
from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    """
    The 12 features describing one house. Field names match the dataset columns
    exactly, because that is what the model pipeline expects.

    - Numeric fields use int with sensible limits (e.g. area must be > 0).
    - Yes/No fields use Literal["yes", "no"]: only those two words are accepted.
    - furnishingstatus only accepts the three known categories.
    """

    area: int = Field(..., gt=0, description="Total area in square feet (must be > 0)")
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=0, le=20, description="Number of bathrooms")
    stories: int = Field(..., ge=0, le=20, description="Number of floors/stories")
    mainroad: Literal["yes", "no"]
    guestroom: Literal["yes", "no"]
    basement: Literal["yes", "no"]
    hotwaterheating: Literal["yes", "no"]
    airconditioning: Literal["yes", "no"]
    parking: int = Field(..., ge=0, le=20, description="Number of parking spaces")
    prefarea: Literal["yes", "no"]
    furnishingstatus: Literal["furnished", "semi-furnished", "unfurnished"]

    # This example shows up pre-filled in the interactive API docs (Swagger UI),
    # so testing the endpoint is one click.
    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """What the API sends back after a successful prediction."""

    predicted_price: float = Field(..., description="Predicted price as a number")
    formatted_price: str = Field(..., description="Human-friendly price, e.g. 'Rs 7,211,633'")
