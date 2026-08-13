"""
main.py
=======
PHASE 8 - The FastAPI application (the web server)

This file defines the API - the set of web addresses (endpoints) the outside
world can call. It deliberately contains NO machine-learning logic: it delegates
all prediction work to model.py, and all data-validation work to schemas.py.

Endpoints:
    GET  /         -> a friendly "API is running" message
    GET  /health   -> a status check ("is the server + model OK?")
    POST /predict  -> receives house features, returns a predicted price

How to run (from the 'backend' folder, with .venv active):
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import HouseFeatures, PredictionResponse
from app.model import model_service


# ---------------------------------------------------------------------------
# STARTUP / SHUTDOWN
# ---------------------------------------------------------------------------
# The 'lifespan' function runs code when the server starts and stops.
# We load the model ONCE at startup so it is ready in memory for every request.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- runs on startup ----
    print("Starting up: loading the trained model...")
    model_service.load()
    print("Model loaded. API is ready to accept requests.")
    yield
    # ---- runs on shutdown ----
    print("Shutting down.")


# Create the FastAPI application object. 'app' is the name uvicorn looks for.
app = FastAPI(
    title="House Price Predictor API",
    description="Predicts house prices using a trained Gradient Boosting model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS  (Cross-Origin Resource Sharing)
# ---------------------------------------------------------------------------
# WHY THIS EXISTS:
# A browser enforces the "same-origin policy": a page loaded from one ORIGIN is
# not allowed to read responses from a different origin. An origin is the
# scheme + host + port, e.g. http://127.0.0.1:5500.
#
# Our project has TWO origins:
#   - the website  (opened as a file://, or served on e.g. http://127.0.0.1:5500)
#   - this API     (http://127.0.0.1:8000)
# The ports differ, so the browser treats them as different origins and would
# BLOCK the fetch() call -- unless the API explicitly says "I allow that origin".
# CORSMiddleware adds the response headers that grant exactly that permission.
#
# DEVELOPMENT vs PRODUCTION:
#   - For local development we allow all origins ("*") so the page works whether
#     you open index.html directly (file://) or serve it on a local port.
#   - In production you should allow ONLY your real website's origin. Swap in the
#     commented-out block below and delete the "*" line.

# --- Development (current): permissive so local testing "just works" ---
ALLOWED_ORIGINS = ["*"]

# --- Production example (use this when you deploy the site for real) ---
# ALLOWED_ORIGINS = [
#     "https://your-website.com",
#     "https://www.your-website.com",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,   # we don't use cookies/logins, so keep this False
    allow_methods=["*"],       # allow GET, POST, and the OPTIONS "preflight" check
    allow_headers=["*"],       # allow Content-Type and any other request headers
)


# ---------------------------------------------------------------------------
# GET /  ->  simple landing message
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    """A basic check that the API is alive."""
    return {"message": "API is running"}


# ---------------------------------------------------------------------------
# GET /health  ->  status check
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """
    Reports whether the server is up AND the model is loaded.
    Useful for confirming everything is ready before sending predictions.
    """
    return {
        "status": "healthy",
        "model_loaded": model_service.is_ready(),
    }


# ---------------------------------------------------------------------------
# POST /predict  ->  the main prediction endpoint
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict_price(house: HouseFeatures):
    """
    Receive house features (already validated by the HouseFeatures schema),
    run them through the model, and return the predicted price.

    FastAPI has ALREADY validated the incoming JSON against HouseFeatures before
    this function runs. So here we can trust the data is complete and well-typed.
    """
    # Make sure the model actually loaded at startup.
    if not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        # Pydantic v2: .model_dump() converts the validated object to a plain dict
        # like {"area": 5000, "bedrooms": 4, ...} - exactly what the model wants.
        features = house.model_dump()
        price = model_service.predict(features)

        return PredictionResponse(
            predicted_price=round(price, 2),
            formatted_price=f"Rs {price:,.0f}",
        )
    except Exception as exc:
        # If anything unexpected goes wrong during prediction, return a clear
        # 500 error instead of crashing the server.
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )
