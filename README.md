# 🏠 House Price Predictor

A supervised **machine-learning web application** that predicts the price of a house (in Indian Rupees, ₹) from its features — area, bedrooms, bathrooms, amenities, and more. A trained **Gradient Boosting** model powers a clean web interface, served through a **FastAPI** backend.

> Predictions are produced by a real trained model at request time — no prices are hard-coded or faked anywhere in the app.

---

## 📖 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Dataset](#-dataset)
6. [Machine-Learning Approach](#-machine-learning-approach)
7. [Model Results](#-model-results)
8. [How It Works](#-how-it-works)
9. [Prerequisites](#-prerequisites)
10. [Installation & Setup](#-installation--setup)
11. [Running the Backend (API)](#-running-the-backend-api)
12. [Running the Frontend (Website)](#-running-the-frontend-website)
13. [API Documentation](#-api-documentation)
14. [Usage](#-usage)
15. [Error Handling](#-error-handling)
16. [CORS](#-cors)
17. [Retraining the Model](#-retraining-the-model)
18. [Troubleshooting](#-troubleshooting)
19. [Future Improvements](#-future-improvements)
20. [Authors & Acknowledgements](#-authors--acknowledgements)
21. [License](#-license)

---

## 🔎 Overview

This project is an end-to-end demonstration of a machine-learning system, built in clearly separated layers:

- **Machine Learning layer** — reads the dataset, builds a preprocessing + model pipeline, compares several regression algorithms, and saves the best one to disk.
- **Backend (API) layer** — a FastAPI server that loads the trained model once at startup and exposes a `/predict` endpoint. It contains **no ML training logic** — it only serves predictions.
- **Frontend layer** — a plain HTML/CSS/JavaScript website that collects house details, sends them to the API, and displays the estimated price.

The three layers are deliberately kept separate so each can be understood, tested, and changed on its own.

---

## ✨ Features

- Predicts house prices from **12 input features** using a trained model.
- Clean, responsive **"valuation certificate"** web interface.
- **FastAPI** backend with automatic, interactive API docs (Swagger UI).
- Strong **input validation** on both the frontend (before sending) and the backend (Pydantic).
- Friendly **error handling** — distinguishes invalid input, an unreachable server, and server-side errors.
- **Reproducible ML pipeline** — preprocessing and the model are saved together in one file, so serving matches training exactly.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data handling | pandas, NumPy |
| Machine learning | scikit-learn (Pipeline, ColumnTransformer, GradientBoostingRegressor) |
| Model persistence | joblib |
| Backend / API | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML, CSS, vanilla JavaScript (`fetch`) |
| Editor | Visual Studio Code |

---

## 📁 Project Structure

```
House-Price-Predictor/
├── backend/
│   ├── app/                     # The FastAPI application (API layer only)
│   │   ├── __init__.py          # Marks 'app' as a Python package
│   │   ├── main.py              # API endpoints (/, /health, /predict) + CORS
│   │   ├── model.py             # Loads the .pkl model, runs predictions
│   │   └── schemas.py           # Request/response validation (Pydantic)
│   ├── data/
│   │   └── Housing.csv          # The dataset (5,000 houses)
│   ├── model/
│   │   └── house_price_model.pkl  # Saved preprocessing + model pipeline
│   ├── data_analysis.py         # Exploratory data analysis (EDA) script
│   ├── preprocessing.py         # Defines features + builds the preprocessor
│   ├── train_model.py           # Trains & compares models, saves the best
│   ├── test_prediction.py       # Quick standalone prediction sanity-check
│   └── requirements.txt         # Python dependencies
│
├── frontend/
│   ├── index.html               # The web page (form + result card)
│   ├── style.css                # Styling
│   └── script.js                # Reads the form, calls the API, shows the price
│
├── .gitignore                   # Files Git should not track (venv, cache, …)
└── README.md                    # This file
```

---

## 📊 Dataset

The model is trained on **`Housing.csv`** — **5,000 houses**, **13 columns** (12 features + the price target).

**Target**

- `price` — the house price in ₹ (a continuous value; this is what we predict).

**Numerical features (5)**

- `area` — total area in square feet
- `bedrooms`, `bathrooms`, `stories` — counts
- `parking` — number of parking spaces

**Binary (yes/no) features (6)**

- `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`

**Categorical feature (1)**

- `furnishingstatus` — one of `furnished`, `semi-furnished`, `unfurnished`

The dataset has **no missing values and no duplicate rows** (verified in `data_analysis.py`).

---

## 🤖 Machine-Learning Approach

**Problem type:** supervised regression (predict a continuous number).

**Preprocessing** (`preprocessing.py`) — built with a scikit-learn `ColumnTransformer` so it can live inside one pipeline:

- Numerical features → `StandardScaler` (puts values on a comparable scale).
- Categorical features → `OneHotEncoder` (turns text categories into 0/1 columns; `handle_unknown="ignore"` so unseen values don't crash prediction).

**Pipeline** — the preprocessor and the model are combined into a single scikit-learn `Pipeline`. This guarantees the **exact same transformations** are applied during training and during live prediction, which removes a whole class of "works in training, breaks in production" bugs.

**Models compared** (`train_model.py`, 80/20 train-test split, `random_state=42`):

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Each model is scored on **MAE**, **RMSE**, and **R²**, and the best is refit on the full dataset and saved to `backend/model/house_price_model.pkl`.

---

## 📈 Model Results

Evaluated on the held-out test set (1,000 houses):

| Model | CV MAE (₹) | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|---:|
| **Linear Regression ✅ (selected)** | **98,626** | **102,249** | **127,679** | **0.9957** |

**Selected model: Linear Regression** — it was selected using 5-fold cross-validation. It explains about **99.6%** of the variation in house prices, with an average error of roughly **₹1.02 lakh** per house on the held-out test set.

> **Data-quality note:** this unusually high score should be interpreted carefully. It may indicate that the uploaded prices follow a near-linear formula based on the available features. Validate the model on independently collected real-world houses before treating the score as market accuracy.

> **Honest note:** this tool gives a data-driven *estimate*, not a guaranteed market price. Accuracy depends on the quality and coverage of the current dataset.

---

## ⚙️ How It Works

```
[ Website form ]  --(house details as JSON)-->  [ FastAPI /predict ]
                                                       |
                                          loads & runs house_price_model.pkl
                                                       |
[ Website shows ₹ price ]  <--(predicted price)--  [ FastAPI response ]
```

1. You enter house details on the web page and click **Estimate price**.
2. JavaScript validates the inputs, then sends them as JSON to `POST /predict`.
3. FastAPI validates the data again (Pydantic), runs it through the saved pipeline, and returns the predicted price.
4. The page displays the price returned by the model.

---

## ✅ Prerequisites

- **Python 3.11** (recommended). *Newer versions such as 3.14 may lack pre-built packages for some libraries.*
- **pip** (comes with Python).
- A modern **web browser** (Chrome, Edge, Firefox, …).
- (Optional) **VS Code** for editing.

Check your Python version:

```powershell
python --version
```

---

## 🛠 Installation & Setup

> Commands below use **Windows PowerShell**. On macOS/Linux, activate the virtual environment with `source venv/bin/activate` instead.

**1. Get the project** and open a terminal in the project's root folder.

**2. Create and activate a virtual environment** (from the `backend` folder):

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Your prompt should now start with `(venv)`.

**3. Install the dependencies:**

```powershell
pip install -r requirements.txt
```

> If the trained model file `backend/model/house_price_model.pkl` is already present, you can skip straight to running the backend. If not, see [Retraining the Model](#-retraining-the-model).

---

## 🚀 Running the Backend (API)

From the `backend` folder, with the virtual environment active:

```powershell
uvicorn app.main:app --reload
```

Expected output:

```
Starting up: loading the trained model...
Model loaded. API is ready to accept requests.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

The API is now live at **http://127.0.0.1:8000**. Leave this window running.

- Interactive API docs (Swagger UI): **http://127.0.0.1:8000/docs**

---

## 🌐 Running the Frontend (Website)

With the backend running, simply open the web page:

- Go to the `frontend` folder and **double-click `index.html`**, or
- (Recommended for development) serve it with a local server such as VS Code's **Live Server** extension.

Then fill in the form and click **Estimate price**.

---

## 📡 API Documentation

Base URL: `http://127.0.0.1:8000`

### `GET /`
Simple check that the API is alive.
```json
{ "message": "API is running" }
```

### `GET /health`
Reports whether the server is up and the model is loaded.
```json
{ "status": "healthy", "model_loaded": true }
```

### `POST /predict`
Predicts a house price from its features.

**Request body (JSON):**
```json
{
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
  "furnishingstatus": "furnished"
}
```

**Successful response (JSON):**
```json
{
  "predicted_price": 7211633.5,
  "formatted_price": "Rs 7,211,633"
}
```

**Field rules (validated by the API):**

| Field | Type | Allowed values |
|---|---|---|
| `area` | integer | greater than 0 |
| `bedrooms`, `bathrooms`, `stories`, `parking` | integer | 0 to 20 |
| `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` | text | `"yes"` or `"no"` |
| `furnishingstatus` | text | `"furnished"`, `"semi-furnished"`, `"unfurnished"` |

---

## 💡 Usage

**Example with `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"area\":5000,\"bedrooms\":4,\"bathrooms\":2,\"stories\":2,\"mainroad\":\"yes\",\"guestroom\":\"no\",\"basement\":\"yes\",\"hotwaterheating\":\"no\",\"airconditioning\":\"yes\",\"parking\":2,\"prefarea\":\"yes\",\"furnishingstatus\":\"furnished\"}"
```

Or just use the website, or the interactive docs at `/docs` (click **Try it out**).

---

## 🛡 Error Handling

**Frontend (`script.js`)** checks inputs *before* sending and shows a clear message for:
- empty fields,
- non-whole or out-of-range numbers,
- an unreachable backend (server not running),
- any error the server reports.

**Backend (FastAPI)** returns clear HTTP errors:
- **422** — invalid or missing fields (automatic Pydantic validation),
- **503** — the model failed to load at startup,
- **500** — an unexpected error during prediction.

---

## 🔗 CORS

Because the website and the API run on different **origins** (different ports), browsers block cross-origin requests by default. The backend enables **CORS** (`CORSMiddleware` in `main.py`) to allow the website to call the API.

- **Development:** all origins are allowed (`"*"`) so the page works whether opened as a file or served locally.
- **Production:** replace `"*"` with your real website's origin(s) — see the commented block in `backend/app/main.py`.

---

## 🔁 Retraining the Model

If you change the dataset or preprocessing, regenerate the model file. From the `backend` folder with the venv active:

```powershell
python train_model.py
```

This compares all models, prints their scores, refits the best one on the full dataset, and overwrites `model/house_price_model.pkl`. To sanity-check the saved model:

```powershell
python test_prediction.py
```

---

## 🩺 Troubleshooting

| Problem | Likely cause & fix |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` / `'fastapi'` | The virtual environment isn't active. Re-run `.\venv\Scripts\Activate.ps1` (prompt should show `(venv)`), then retry. |
| `Activate.ps1 cannot be loaded` (execution policy) | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `pip install` fails to build packages | You're likely on a very new Python (e.g. 3.14) without pre-built wheels. Use **Python 3.11**. |
| Website shows "Could not reach the prediction server" | The backend isn't running. Start it with `uvicorn app.main:app --reload` and confirm `Application startup complete`. |
| Price won't update after code changes | Hard-refresh the browser (`Ctrl + F5`) so the new JavaScript loads. |
| `Model file not found` on startup | Run `python train_model.py` to create `model/house_price_model.pkl`. |

---

## 🚧 Future Improvements

- Add more features (location, age of property, lot size) to improve accuracy.
- Try more powerful models (XGBoost, LightGBM) and hyperparameter tuning.
- Show a confidence range alongside the point estimate.
- Deploy the API and website online.
- Add automated tests for the API.

---

## 👥 Authors & Acknowledgements

- Built as a **Student Group Project (SGP)** by **HP** and team.
- Dataset: the public **Housing** dataset (`Housing.csv`).
- Built with open-source tools: scikit-learn, FastAPI, and Uvicorn.

---

## 📄 License

This project was created for educational purposes as part of a student group project. You are free to study, use, and adapt it for learning.
