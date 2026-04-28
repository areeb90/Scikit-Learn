# 🔍 Titanic Survival Predictor — SHAP Explainability

## What This Is

This module adds **individual prediction explanations** to the trained Titanic model using SHAP
(SHapley Additive exPlanations). For any passenger, you can see exactly how much each feature
pushed the model's prediction toward survival or away from it — down to the exact probability
contribution of each variable.

---

## The Problem It Solves

A trained Random Forest can tell you: *"this passenger has a 93% chance of survival."*

But it cannot tell you *why* — not without SHAP.

This matters in two contexts:

**In interviews:** Most candidates stop at accuracy metrics. Being able to say "I can explain any
individual prediction and quantify each feature's contribution" signals a level of understanding
that is rare at the junior level.

**In production:** ML models that make decisions affecting people (credit, hiring, healthcare) are
increasingly required by law or company policy to be explainable. A model you cannot explain is a
model you cannot deploy in regulated environments.

SHAP solves both.

---

## The Core Idea — What SHAP Actually Calculates

SHAP comes from cooperative game theory. The question it answers is:

> "If multiple features are working together to produce a prediction, how do we fairly attribute
> credit (or blame) to each one?"

For every prediction, SHAP computes a **SHAP value** for each feature. This is a signed number:

- **Positive SHAP value** → this feature pushed the prediction *toward* survival
- **Negative SHAP value** → this feature pushed the prediction *away* from survival
- **Magnitude** → how much it pushed (a value of 0.3 is a stronger push than 0.05)

The SHAP values for all features sum to a number that, added to the **base value** (the model's
average prediction across all training data), gives you the model's exact output for that passenger.
Every percentage point in the final probability is fully accounted for.

```
Base value (average prediction across all passengers)
    + SHAP(sex_female)
    + SHAP(pclass)
    + SHAP(fare)
    + SHAP(age)
    + SHAP(title_Miss)
    + ... (all other features)
    ─────────────────────────
    = Final survival probability for this passenger
```

This is what makes SHAP different from feature importance. Feature importance tells you the average
effect across all predictions. SHAP tells you the exact effect for *this specific passenger*.

---

## The Pipeline Challenge and How It Was Solved

The Titanic model is a **sklearn Pipeline**:

```
Raw DataFrame → ColumnTransformer → RandomForestClassifier
```

SHAP's `TreeExplainer` works on the `RandomForestClassifier` directly — it does not understand the
full Pipeline. This creates a challenge: the explainer needs data in the form the classifier sees
(after preprocessing), not the raw DataFrame.

The solution is a two-step process:

```python
# Step 1: Extract the fitted preprocessor from the pipeline
preprocessor_fitted = best_rf.named_steps['preprocessor']
rf_model            = best_rf.named_steps['model']

# Step 2: Transform X_test through the preprocessor only
X_test_transformed = preprocessor_fitted.transform(X_test)

# Step 3: Wrap in DataFrame with real feature names for readable plots
X_test_shap = pd.DataFrame(X_test_transformed, columns=all_feature_names)

# Step 4: Point TreeExplainer at the classifier, not the pipeline
explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_shap)
```

This is the correct pattern for using SHAP with any sklearn Pipeline that contains a tree-based
model. It guarantees that the data SHAP sees is identical to the data the classifier was trained on.

---

## Feature Names After Encoding

After the `ColumnTransformer` processes the data, the 10 original columns become 17 features
(because One-Hot Encoding expands categorical columns into binary columns):

**Numeric features (unchanged by preprocessing, 7 total):**
```
pclass, age, sibsp, parch, fare, family_size, is_alone
```

**Categorical features after One-Hot Encoding (10 total):**
```
sex_female, sex_male
embarked_C, embarked_Q, embarked_S
title_Master, title_Miss, title_Mr, title_Mrs, title_Rare
```

SHAP computes a value for each of these 17 features for every prediction.

---

## The Three Plots

### Plot 1: Beeswarm Plot — Global Feature Impact

![SHAP Beeswarm](shap_beeswarm.png)

**What it shows:** Every dot represents one test passenger. The horizontal position shows how much
that feature pushed the model's prediction. The color shows the feature value — red means high
value, blue means low value.

**How to read it:**
- Features are sorted by total impact (most impactful at the top)
- A cluster of red dots on the right side of a row means: high values of this feature push toward
  survival
- A cluster of blue dots on the left means: low values push away from survival

**What the Titanic data reveals:**
- `sex_female`: red dots (female) cluster strongly on the right → being female dramatically
  increases survival probability. This is the "women and children first" policy visible in data.
- `pclass`: blue dots (low class = high class number) cluster on the left → third class passengers
  were much less likely to survive
- `fare`: red dots (high fare) on the right → expensive tickets correlate with survival, partly
  because first class passengers had better access to lifeboats

---

### Plot 2: Waterfall Plot — Individual Prediction Breakdown

![SHAP Waterfall - Survived](shap_waterfall_survived.png)
![SHAP Waterfall - Not Survived](shap_waterfall_not_survived.png)

**What it shows:** A complete breakdown of one specific prediction. Every bar represents one
feature's contribution. The chart starts at the base value (average prediction) and each bar
moves it up or down until the final prediction is reached.

**How to read it:**
- The bottom shows `E[f(x)]` — the model's average predicted survival probability across all
  training passengers (the starting point)
- Red bars push the probability upward (toward survival)
- Blue bars push the probability downward (toward death)
- The top shows `f(x)` — this passenger's final predicted probability
- The numbers on each bar show the exact SHAP value for that feature

**Example — a first-class female passenger:**
```
Base value:     0.38  (average prediction)
sex_female:    +0.31  (being female, strong push toward survival)
pclass:        +0.09  (first class, moderate push up)
fare:          +0.08  (high fare, moderate push up)
age:           -0.02  (slightly older, small push down)
is_alone:      -0.01  (travelling alone, tiny push down)
─────────────────────
Final:          0.93  (93% survival probability)
```

Every single percentage point is explained.

---

### Plot 3: Bar Chart — Clean Feature Importance for README

![SHAP Bar](shap_bar.png)

**What it shows:** The mean absolute SHAP value for each feature across all test passengers. This
is a cleaner, simpler view of global feature importance — which features mattered most on average.

**Difference from sklearn feature importance:** sklearn's `feature_importances_` uses impurity
reduction — how much each feature reduced uncertainty across all splits in all trees. SHAP's bar
chart uses the actual impact on output values. SHAP is considered more reliable because it directly
measures the effect on predictions rather than tree structure.

---

## SHAP Version Note — 3D Array Format

Different versions of SHAP return results in different formats for binary classification:

- **Older SHAP (< 0.40):** returns a Python list of two arrays: `[array_class_0, array_class_1]`
  — access with `shap_values[1]`
- **Newer SHAP (>= 0.40):** returns a single 3D numpy array of shape `(n_samples, n_features, 2)`
  — access with `shap_values[:, :, 1]`

To check which format you have:
```python
print(type(shap_values))        # list → old format, ndarray → new format
print(shap_values[1].shape)     # if (n_samples, n_features) → old format
                                # if (n_features, 2) → new format, use [:, :, 1]
```

In this project, SHAP returned the 3D array format. The correct slice for the Survived class is:
```python
shap_vals_survived = shap_values[:, :, 1]   # shape: (262, 17)
```

---

## Key Results

| Metric | Value |
|---|---|
| Test set size | 262 passengers |
| Features after encoding | 17 |
| Base value (average prediction) | ~0.38 |
| Most impactful feature | `sex_female` |
| Second most impactful | `pclass` |
| Model test ROC-AUC | 0.8951 |

---

## What You Can Say in an Interview

**Without SHAP:**
> "My model achieved ROC-AUC 0.89 on the test set."

**With SHAP:**
> "My model achieved ROC-AUC 0.89. I added SHAP explainability so I can account for every
> percentage point in any prediction. For example, for a third-class male passenger, `sex_male`
> pushed the survival probability down by 0.28 and `pclass` pushed it down by another 0.14,
> bringing the final probability to around 0.07. Every prediction is fully auditable."

The second answer demonstrates that you understand what your model actually learned — not just
that it performs well, but *why* it makes the decisions it makes.

---

## Files Generated

| File | Description |
|---|---|
| `shap_beeswarm.png` | Global feature impact — all 262 test passengers |
| `shap_bar.png` | Mean absolute SHAP value per feature — clean README chart |
| `shap_waterfall_survived.png` | Full prediction breakdown for a surviving passenger |
| `shap_waterfall_not_survived.png` | Full prediction breakdown for a non-surviving passenger |

---

## Resume Bullet

> Integrated SHAP TreeExplainer with the sklearn Pipeline to produce individual prediction
> explanations across 262 test passengers — waterfall plots decompose each prediction into
> per-feature probability contributions, enabling full model auditability and bias detection.
