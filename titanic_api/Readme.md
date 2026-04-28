



<img width="1685" height="1521" alt="image" src="https://github.com/user-attachments/assets/c9593e57-3d95-450c-bc61-3bc606c964bc" />

<img width="1681" height="1404" alt="image" src="https://github.com/user-attachments/assets/bf35ab32-228a-4c23-8f3a-f6604ac81c1b" />

<img width="1664" height="1507" alt="image" src="https://github.com/user-attachments/assets/8c7ff211-c172-4de7-aefe-5c4ebbd2d66a" />

<img width="1676" height="1453" alt="image" src="https://github.com/user-attachments/assets/1264f95b-e66d-48ab-857a-f71cf71da5d7" />


=================
PYTHON REQUESTS
=================

<img width="1770" height="1160" alt="image" src="https://github.com/user-attachments/assets/7832281b-75b8-4683-8b24-9c1efe88f6a2" />


# 🚀 Titanic Survival Predictor — FastAPI Serving

## What This Is

This module wraps the trained Titanic survival model in a production-style REST API using FastAPI.
Instead of running a notebook to get a prediction, any application — a website, a mobile app, another
service — can send an HTTP request and get back a survival probability in milliseconds.

This is the difference between a **notebook model** and a **deployed model**.

---

## The Problem It Solves

After training, a model lives inside a `.joblib` file. It is completely inaccessible to anything
outside Python. A trained model that cannot be queried is not useful in production.

FastAPI solves this by exposing the model through a standard web interface. Any system that can make
an HTTP request — which is everything — can now use your model.

---

## How It Works — The Full Picture

```
User / Application
       │
       │  POST /predict
       │  JSON body: { pclass, age, fare, sex, ... }
       ▼
┌─────────────────────┐
│     FastAPI App      │
│                     │
│  1. Pydantic parses │
│     and validates   │
│     the input       │
│                     │
│  2. Builds a pandas │
│     DataFrame       │
│                     │
│  3. Passes it to    │
│     the sklearn     │
│     Pipeline        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  titanic_model      │
│  .joblib            │
│                     │
│  ColumnTransformer  │
│  ├── num_pipeline   │
│  │   ├── Imputer    │
│  │   └── Scaler     │
│  └── cat_pipeline   │
│      ├── Imputer    │
│      └── OHE        │
│                     │
│  RandomForest       │
│  (best from Grid    │
│   SearchCV)         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  JSON Response      │
│  {                  │
│   prediction: 1,    │
│   label: Survived,  │
│   probability: 0.93 │
│  }                  │
└─────────────────────┘
```

The critical insight: **the sklearn Pipeline handles all preprocessing internally**. The API does not
manually impute, scale, or encode anything. It simply builds a DataFrame with the right column names
and passes it to `model.predict()`. The Pipeline takes care of everything else — exactly as it did
during training.

---

## Project Structure

```
titanic_api/
├── main.py                  ← the FastAPI application
├── titanic_model.joblib     ← trained sklearn Pipeline (saved in Section 9)
└── requirements.txt         ← dependencies
```

---

## What the Model Expects

The model was trained on these exact features (from the Titanic feature engineering in Section 4):

| Feature | Type | Description |
|---|---|---|
| `pclass` | int | Passenger class: 1, 2, or 3 |
| `age` | float | Age in years |
| `sibsp` | int | Number of siblings/spouses aboard |
| `parch` | int | Number of parents/children aboard |
| `fare` | float | Ticket price in GBP |
| `sex` | str | `'male'` or `'female'` |
| `embarked` | str | Port of embarkation: `'C'`, `'Q'`, or `'S'` |
| `family_size` | int | `sibsp + parch + 1` (engineered feature) |
| `is_alone` | int | `1` if travelling alone, `0` otherwise (engineered feature) |
| `title` | str | Extracted from name: `'Mr'`, `'Mrs'`, `'Miss'`, `'Master'`, `'Rare'` |

`family_size` and `is_alone` are **engineered features** created in Section 4 of the notebook.
They must be computed before sending a request — the API does not derive them automatically.

---

## API Endpoints

### `GET /health`
Returns the server status and model type. Use this to check the API is running before sending
prediction requests.

**Response:**
```json
{
  "status": "ok",
  "model": "Pipeline"
}
```

---

### `POST /predict`
Accepts a passenger's features and returns a survival prediction with probability.

**Request body:**
```json
{
  "pclass": 1,
  "age": 29.0,
  "sibsp": 0,
  "parch": 0,
  "fare": 211.34,
  "sex": "female",
  "embarked": "S",
  "family_size": 1,
  "is_alone": 1,
  "title": "Miss"
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Survived",
  "survival_probability": 0.93
}
```

**Fields:**
- `prediction` — `1` for survived, `0` for did not survive
- `prediction_label` — human-readable version of the prediction
- `survival_probability` — the model's confidence that this passenger survived (0.0 to 1.0)

---

### `GET /docs`
FastAPI auto-generates interactive Swagger documentation. Open this in a browser to explore and
test all endpoints without writing any code.

---

## Running the API

**Install dependencies:**
```bash
pip install fastapi uvicorn joblib scikit-learn pandas numpy
```

**Start the server:**
```bash
uvicorn main:app --reload
```

The `--reload` flag restarts the server automatically every time `main.py` is saved. Leave this
running in one terminal while you test in another.

**Verify it's running:**

Open `http://127.0.0.1:8000/health` in your browser. You should see:
```json
{"status": "ok", "model": "Pipeline"}
```

---

## Testing

### Option 1 — Swagger UI (recommended, no code needed)
Open `http://127.0.0.1:8000/docs` in your browser. Click **POST /predict → Try it out → Execute**.
Fill in the passenger fields and see the response live.

### Option 2 — PowerShell
```powershell
$body = @{
    pclass=1; age=29.0; sibsp=0; parch=0; fare=211.34
    sex="female"; embarked="S"; family_size=1; is_alone=1; title="Miss"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
                  -Method POST `
                  -Body $body `
                  -ContentType "application/json"
```

### Option 3 — Python requests
```python
import requests

payload = {
    "pclass": 3, "age": 22.0, "sibsp": 0, "parch": 0, "fare": 7.25,
    "sex": "male", "embarked": "S", "family_size": 1, "is_alone": 1, "title": "Mr"
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(response.json())
# {"prediction": 0, "prediction_label": "Did not survive", "survival_probability": 0.07}
```

---

## Input Validation

FastAPI uses **Pydantic** for automatic input validation. If a request is missing a field or sends
the wrong type, the API returns a clear error — it never crashes silently.

Example — missing `age` field:
```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

This means consumers of the API always get actionable error messages, not stack traces.

---

## Why This Matters

| Before FastAPI | After FastAPI |
|---|---|
| Model only works inside a Jupyter notebook | Model accessible to any application via HTTP |
| Predictions require Python + sklearn installed | Any language (JS, Java, R, curl) can call it |
| No input validation — wrong inputs cause crashes | Pydantic validates every field before it touches the model |
| Sharing results means sharing the whole notebook | Share one URL |

---

## Key Design Decisions

**Model loaded once at startup, not per request.** Loading a `.joblib` file takes ~100ms. If the
model were loaded on every request, a server handling 100 requests/second would spend all its time
loading the model rather than running inference. Loading at startup means the model is always in
memory and predictions return in milliseconds.

**Pydantic schema matches training columns exactly.** The field names in `PassengerFeatures` are
identical to the column names the `ColumnTransformer` was trained on. If they don't match, the
Pipeline raises a `ValueError`. The schema is the contract between the API and the model.

**The Pipeline handles preprocessing.** The API does not impute, scale, or encode anything manually.
This is intentional — the fitted Pipeline from training already knows the median values for
imputation and the scaler parameters. Recomputing these in the API would introduce inconsistencies
between training and serving.

---

## Resume Bullet

> Built a FastAPI REST endpoint serving a tuned scikit-learn Pipeline (GridSearchCV, ROC-AUC 0.8951)
> with Pydantic input validation, automatic Swagger documentation, and sub-millisecond inference —
> moving the model from notebook to production-accessible service.
