### Titanic Survival Predictor

I built an end-to-end ML system on the Titanic dataset. The goal was not just to train a model — it was to build something deployable. I engineered features from raw passenger data, built a sklearn Pipeline that handles all preprocessing automatically, tuned hyperparameters with GridSearchCV, and reached ROC-AUC 0.8951 on the test set.
I then added three production layers: SHAP explainability so any prediction can be fully audited, MLflow tracking so every experiment is logged and reproducible, and a FastAPI endpoint so the model is accessible via HTTP to any application.
The key finding was that gender and passenger class were the dominant survival predictors — SHAP quantified this exactly. Being female pushed survival probability up by 0.31 on average. Being third class pushed it down by 0.14. The model learned structural inequalities from data alone.
