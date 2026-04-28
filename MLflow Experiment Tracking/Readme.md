<img width="2344" height="1476" alt="image" src="https://github.com/user-attachments/assets/6fa3bebc-080f-4ea7-b247-989920a580d1" />

<img width="2331" height="1521" alt="image" src="https://github.com/user-attachments/assets/e994854e-28ff-45e6-8cc8-e4b4872d33dd" />

<img width="2351" height="1491" alt="image" src="https://github.com/user-attachments/assets/95d63db9-4cad-4962-8971-b7232a7d0569" />

<img width="2332" height="1352" alt="image" src="https://github.com/user-attachments/assets/22996978-a075-41f6-ba63-a6a99f561aba" />

<img width="2320" height="1446" alt="image" src="https://github.com/user-attachments/assets/9aa0b9e2-a2c1-4126-81d2-af270bc5abfa" />



# 📊 Titanic Survival Predictor — MLflow Experiment Tracking

## What This Is

This module integrates **MLflow experiment tracking** into the Titanic model training pipeline.
Every training run — every model, every hyperparameter combination, every evaluation metric — is
logged to a persistent database and viewable in a browser-based UI. No results are ever lost to
a cleared print statement again.

---

## The Problem It Solves

Consider what happens without experiment tracking during a week of model development:

```
Monday:   RF n_estimators=100  →  printed "ROC-AUC: 0.87"  →  terminal closed
Tuesday:  RF n_estimators=200  →  printed "ROC-AUC: 0.89"  →  terminal closed
Wednesday: max_depth=5          →  printed "ROC-AUC: 0.91"  →  terminal closed
Thursday: max_depth=10          →  printed "ROC-AUC: 0.88"  →  terminal closed
Friday:   Manager asks: "which model are we deploying and what settings produced it?"
```

You have no answer. The prints are gone. You cannot reproduce the best result without rerunning
everything from scratch. You cannot prove which hyperparameters produced which metric.

This is not a hypothetical — it is the standard experience of every data scientist who has
worked without tracking. MLflow eliminates this problem entirely.

---

## The Core Concept — What MLflow Actually Stores

Every time you call `mlflow.start_run()`, MLflow creates a **run** — a permanent record containing:

```
Run: RandomForest-BestModel-Final
├── Parameters (what you configured)
│   ├── n_estimators     = 200
│   ├── max_depth        = 5
│   ├── min_samples_leaf = 1
│   ├── cv_folds         = 5
│   └── scoring          = roc_auc
│
├── Metrics (what you measured)
│   ├── test_roc_auc     = 0.8951
│   ├── test_accuracy    = 0.8473
│   ├── cv_roc_auc_mean  = 0.8923
│   └── cv_roc_auc_std   = 0.0187
│
├── Tags (labels you set)
│   ├── model_type       = RandomForest
│   ├── stage            = production_candidate
│   └── dataset          = titanic-openml
│
└── Artifacts (files you saved)
    └── titanic-rf-pipeline/
        ├── model.pkl        ← the actual sklearn Pipeline
        ├── MLmodel          ← MLflow metadata
        ├── conda.yaml       ← environment specification
        └── requirements.txt ← Python dependencies
```

Every run is stored permanently. You can query, compare, and load any of them at any time —
days, weeks, or months later.

---

## How It Fits Into This Project

MLflow was integrated at three levels:

### Level 1 — Baseline model comparison
Four models were logged as separate runs under the same experiment:

| Run name | test_roc_auc | test_accuracy |
|---|---|---|
| LogisticRegression | 0.8830 | — |
| RandomForest | 0.8534 | — |
| GradientBoosting | 0.8938 | — |
| SVM | 0.8656 | — |

Each run took ~3 lines to add to existing training code:
```python
with mlflow.start_run(run_name="RandomForest"):
    mlflow.log_metric("test_roc_auc", test_roc_auc)
    mlflow.log_metric("test_accuracy", test_acc)
```

### Level 2 — GridSearchCV: every candidate logged
The `GridSearchCV` search space had 18 combinations
(`n_estimators` × `max_depth` × `min_samples_leaf` = 2 × 3 × 3). Every candidate was logged
as a **nested child run** under a parent run, so the full search history is preserved:

```
RandomForest-GridSearch-Parent (parent run)
├── candidate_00  n_estimators=100, max_depth=None, min_samples_leaf=1
├── candidate_01  n_estimators=100, max_depth=None, min_samples_leaf=2
├── candidate_02  n_estimators=100, max_depth=None, min_samples_leaf=4
├── candidate_03  n_estimators=100, max_depth=5,    min_samples_leaf=1
│   ... (18 candidates total)
└── candidate_17  n_estimators=200, max_depth=10,   min_samples_leaf=4
```

This answers a question that print statements cannot: *"did increasing `n_estimators` from 100
to 200 actually help, and by how much?"* — you can answer this by sorting candidates by
`cv_roc_auc_mean` in the UI.

### Level 3 — Final model as a registered artifact
The best model was logged as a complete artifact — not just metrics, but the actual fitted
sklearn Pipeline serialised and stored by MLflow:

```python
mlflow.sklearn.log_model(
    sk_model      = grid_search.best_estimator_,
    artifact_path = "titanic-rf-pipeline"
)
```

This artifact contains everything needed to serve the model: the fitted `ColumnTransformer`
(with imputation medians and OHE categories from training data) and the fitted
`RandomForestClassifier`. Loading it produces identical predictions to the original notebook.

---

## Experiment Structure

```
Experiment: titanic-survival-prediction
│
├── LogisticRegression              (baseline)
├── RandomForest                    (baseline)
├── GradientBoosting                (baseline)
├── SVM                             (baseline)
│
├── RandomForest-GridSearch-Parent  (tuning)
│   ├── candidate_00
│   ├── candidate_01
│   │   ... 18 candidates
│   └── candidate_17
│
└── RandomForest-BestModel-Final    (production candidate)
    ├── params:   n_estimators=200, max_depth=5, min_samples_leaf=1
    ├── metrics:  test_roc_auc=0.8951, test_accuracy=0.8473
    └── artifact: titanic-rf-pipeline/
```

---

## Running the UI

```bash
cd "C:\101 Mahcine Learning\Projects\Scikit-Learn\MLflow Experiment Tracking"
mlflow ui --backend-store-uri "file:///C:/101 Mahcine Learning/Projects/Scikit-Learn/MLflow Experiment Tracking/mlruns"
```

Open `http://127.0.0.1:5000` in your browser.

**Important:** The tracking URI in the notebook and the path passed to `mlflow ui` must point
to the same location. If they differ, the UI shows an empty experiment. Always set the tracking
URI explicitly at the top of any notebook that logs to MLflow:

```python
mlflow.set_tracking_uri(
    r"file:///C:/101 Mahcine Learning/Projects/Scikit-Learn/MLflow Experiment Tracking/mlruns"
)
```

---

## The UI — What Each Section Shows You

### Experiments list (left sidebar)
All experiments grouped by name. An experiment is a collection of related runs — all Titanic
runs live under `titanic-survival-prediction`.

### Training Runs table
Every run as a row. Columns are your logged params and metrics. **Sort by `test_roc_auc`
descending and the best model floats to the top immediately.** This is the primary workflow
for selecting a model to deploy.

### Run detail page
Click any run to see its complete record: every param, every metric, every tag, and a
downloadable link to its artifact. This page is the permanent receipt for that training run.

### Run comparison
Select two or more runs and click **Compare**. MLflow renders a side-by-side diff of all params
and metrics. This directly answers questions like:

- Did `max_depth=5` outperform `max_depth=None`?
- Did `GradientBoosting` beat `RandomForest` on ROC-AUC, or only on accuracy?
- Which GridSearch candidate had the lowest `cv_roc_auc_std` (most stable)?

---

## Loading a Logged Model

A model artifact logged to MLflow can be loaded back and used for inference without the original
notebook or training code:

```python
import mlflow.sklearn

# Load by run ID (always works)
run_id = "2b193c1d8f984e7eb7d21f77924cd918"
loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/titanic-rf-pipeline")

# Use identically to the original pipeline
prediction = loaded_model.predict(X_test.iloc[[0]])
probability = loaded_model.predict_proba(X_test.iloc[[0]])[0][1]

print(f"Prediction  : {'Survived' if prediction[0] == 1 else 'Did not survive'}")
print(f"Probability : {probability:.2%}")
# Prediction  : Survived
# Probability : 60.25%
```

This is significant: the model can be loaded in a completely separate Python session, on a
different machine, without access to the original training data — and it produces identical
predictions. This is what "reproducible ML" means in practice.

---

## Key Design Decisions

**Why nested runs for GridSearchCV candidates?**
Nesting keeps the UI clean. Without nesting, 18 candidates and 4 baselines would all appear at
the same level, making it hard to distinguish tuning runs from baseline comparisons. Nested runs
group the 18 candidates under their parent, so the top-level view shows only meaningful runs.

**Why log `cv_roc_auc_std` in addition to `cv_roc_auc_mean`?**
A model with mean ROC-AUC 0.88 and std 0.01 is more reliable than one with mean 0.89 and std
0.06. The std tells you whether the model's performance is consistent across different subsets
of the training data. Logging both gives a complete picture of model stability.

**Why use `file://` URI instead of SQLite for the tracking store?**
The file-based backend requires no database setup — MLflow creates a folder structure in `mlruns/`
automatically. SQLite is slightly faster for large numbers of runs but requires correct URI
formatting and can cause path issues on Windows. For a portfolio project, file-based is simpler
and equally functional.

---

## Connecting MLflow to FastAPI

The full production pattern connects the two systems: MLflow stores the model, FastAPI serves it.
Instead of loading `titanic_model.joblib`, the API loads directly from the MLflow run:

```python
import mlflow.sklearn

# In main.py — load model from MLflow instead of joblib
RUN_ID = "2b193c1d8f984e7eb7d21f77924cd918"
model  = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/titanic-rf-pipeline")
```

This means: to deploy a new model version, you update one string (the run ID) in the API — no
file copying, no manual joblib exports. The model registry is the single source of truth.

---

## MLflow Concepts Glossary

| Term | What it means in this project |
|---|---|
| **Tracking URI** | Where MLflow writes data — a folder path or database connection |
| **Experiment** | A named collection of runs — `titanic-survival-prediction` |
| **Run** | One training execution — one set of params + metrics + artifacts |
| **Parameter** | A configuration value you set before training (`n_estimators=200`) |
| **Metric** | A number you measure after training (`test_roc_auc=0.8951`) |
| **Tag** | A label you attach to a run (`stage=production_candidate`) |
| **Artifact** | A file logged to a run — the saved sklearn Pipeline |
| **Nested run** | A child run grouped under a parent — used for GridSearch candidates |

---

## Resume Bullet

> Integrated MLflow experiment tracking across 4 baseline models and 18 GridSearchCV candidates
> — logged hyperparameters, ROC-AUC and accuracy metrics, and the complete sklearn Pipeline as
> a reloadable artifact, enabling reproducible model comparison and one-line model loading for
> deployment (test ROC-AUC: 0.8951, test accuracy: 0.8473).
