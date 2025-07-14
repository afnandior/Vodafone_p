# 1 Import libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
import lightgbm as lgb

# 2 Preprocessing
def preprocess_data(df, features, target):
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df = df.dropna(subset=[target] + features)
    X = df[features]
    y = df[target]
    return X, y

# 3 Split Data
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 4 Define Models
def define_models():
    models = {
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
        "Lasso Regression": Pipeline([("scaler", StandardScaler()), ("model", Lasso())]),
        "ElasticNet Regression": Pipeline([("scaler", StandardScaler()), ("model", ElasticNet())]),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": xgb.XGBRegressor(),
        "LightGBM": lgb.LGBMRegressor(),
        "Polynomial (deg=2)": Pipeline([("poly", PolynomialFeatures(degree=2)), ("scaler", StandardScaler()), ("linear", LinearRegression())]),
        "SVR": Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
        "KNN Regression": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
        "Neural Network (MLPRegressor)": Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(max_iter=1000))]),
    }
    return models

# 5 Define Hyperparameters
def define_param_grids():
    param_grids = {
        "Polynomial (deg=2)": {"poly__degree": [2, 3]},
        "Ridge Regression": {"model__alpha": [0.1, 1.0, 10.0]},
        "Lasso Regression": {"model__alpha": [0.01, 0.1, 1.0]},
        "ElasticNet Regression": {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]},
        "Decision Tree": {"max_depth": [3, 5, 10]},
        "Random Forest": {"n_estimators": [50, 100, 200]},
        "Gradient Boosting": {"n_estimators": [50, 100, 200]},
        "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        "LightGBM": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        "SVR": {"model__kernel": ["linear", "rbf"], "model__C": [0.1, 1, 10]},
        "KNN Regression": {"model__n_neighbors": [3, 5, 10]},
        "Neural Network (MLPRegressor)": {"model__hidden_layer_sizes": [(50,), (100,), (100,50)], "model__activation": ["relu", "tanh"]},
    }
    return param_grids

# 6 Train & Tune Models
def train_and_tune_models(models, X_train, X_test, y_train, y_test, param_grids, manual_hyperparams=None):
    results = []
    best_r2 = -np.inf
    best_model_name = None
    best_model = None
    best_params = None
    tuned_model = None

    for name, model in models.items():
        try:
            print(f"\n Training {name} ...")

            # Apply manual hyperparameters if provided
            if manual_hyperparams and name in manual_hyperparams:
                print(f" Applying manual hyperparameters for {name}: {manual_hyperparams[name]}")
                model.set_params(**manual_hyperparams[name])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

            this_best_params = manual_hyperparams.get(name) if manual_hyperparams else None
            this_tuned_r2 = None
            this_tuned_model = None

            # Perform GridSearchCV tuning if no manual hyperparameters
            if not (manual_hyperparams and name in manual_hyperparams) and name in param_grids:
                print(f" Performing GridSearchCV for {name} ...")
                grid = GridSearchCV(model, param_grids[name], cv=5, scoring="r2")
                grid.fit(X_train, y_train)
                this_tuned_model = grid.best_estimator_
                this_tuned_r2 = grid.best_score_
                this_best_params = grid.best_params_

            # Update best model
            if cv_r2 > best_r2:
                best_r2 = cv_r2
                best_model_name = name
                best_model = model
                tuned_model = this_tuned_model
                best_params = this_best_params

            results.append({
                "Model": name,
                "Test_R2": r2,
                "CV_R2": cv_r2,
                "Test_MSE": mse,
                "Best_Params": this_best_params,
                "Tuned_CV_R2": this_tuned_r2
            })
        except Exception as e:
            print(f" Error in model {name}: {e}")

    results_df = pd.DataFrame(results).sort_values(by="CV_R2", ascending=False).reset_index(drop=True)
    return results_df, best_model_name, best_model, tuned_model, best_params

# 7 Main pipeline
def regression_pipeline_full(df, features, target, model_save_path="best_model.pkl", manual_hyperparams=None, save_model=True):
    X, y = preprocess_data(df, features, target)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = define_models()
    param_grids = define_param_grids()

    results_df, best_model_name, best_model, tuned_model, best_params = train_and_tune_models(
        models, X_train, X_test, y_train, y_test, param_grids, manual_hyperparams
    )

    if save_model:
        joblib.dump(best_model, model_save_path)
        print(f"\n Best model saved as {model_save_path}")
    else:
        print("\nℹ️ Model not saved (save_model=False)")

    return results_df, best_model_name, best_model, tuned_model, best_params

# 8 Prediction function
def predict_new_data(model, new_data_df, features):
    X_new = new_data_df[features]
    predictions = model.predict(X_new)
    return predictions.round(2)

# 9 Load model
def load_model(path="best_model.pkl"):
    return joblib.load(path)

