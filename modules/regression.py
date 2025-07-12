# Import libraries

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


# 1 Preprocessing
def preprocess_data(df, features, target):
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df = df.dropna(subset=[target] + features)
    X = df[features]
    y = df[target]
    return X, y


# 2 Split Data
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# 3 Define Models
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


# 4 Define Hyperparameters
def define_param_grids():
    param_grids = {
        "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        "LightGBM": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        "Ridge Regression": {"model__alpha": [0.1, 1.0, 10.0]},
        "Lasso Regression": {"model__alpha": [0.01, 0.1, 1.0]},
        "ElasticNet Regression": {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]},
        "Decision Tree": {"max_depth": [3, 5, 10]},
        "Random Forest": {"n_estimators": [50, 100, 200]},
        "Gradient Boosting": {"n_estimators": [50, 100, 200]},
        "SVR": {"model__kernel": ["linear", "rbf"], "model__C": [0.1, 1, 10]},
        "KNN Regression": {"model__n_neighbors": [3, 5, 10]},
        "Neural Network (MLPRegressor)": {"model__hidden_layer_sizes": [(50,), (100,), (100,50)], "model__activation": ["relu", "tanh"]},
    }
    return param_grids


# 5 Train & Tune Models
def train_and_tune_models(models, X_train, X_test, y_train, y_test, param_grids):
    results = []
    best_r2 = -np.inf
    best_model_name = None
    best_model = None
    best_params = None
    tuned_model = None

    for name, model in models.items():
        try:
            print(f" Training {name} ...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

            this_best_params = None
            this_tuned_r2 = None
            this_tuned_model = None

            if name in param_grids:
                print(f"   Performing GridSearchCV for {name} ...")
                grid = GridSearchCV(model, param_grids[name], cv=5, scoring="r2")
                grid.fit(X_train, y_train)
                this_tuned_model = grid.best_estimator_
                this_tuned_r2 = grid.best_score_
                this_best_params = grid.best_params_

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
            print(f"Error in model {name}: {e}")

    results_df = pd.DataFrame(results).sort_values(by="CV_R2", ascending=False)
    return results_df, best_model_name, best_model, tuned_model, best_params


# 6 Tune specific model
def tune_model(model, X_train, y_train, param_grid):
    grid = GridSearchCV(model, param_grid, cv=5, scoring="r2")
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


# 7 Main pipeline
def regression_pipeline_full(df, features, target, test_size=0.2, random_state=42, param_grids=None, manual_params=None):
    X, y = preprocess_data(df, features, target)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    models = define_models()
    default_param_grids = define_param_grids()
    if param_grids is None:
        param_grids = default_param_grids

    results_df, best_model_name, best_model, tuned_model, best_params = train_and_tune_models(
        models, X_train, X_test, y_train, y_test, param_grids
    )

    if manual_params:
        print(f"\n Using Manual Hyperparameters for {best_model_name}: {manual_params}")
        tuned_model, best_params = tune_model(best_model, X_train, y_train, manual_params)
        print(f" Manual tuning completed. Best Params: {best_params}")

    joblib.dump(best_model, "best_model.pkl")
    print(" Best model saved as best_model.pkl")

    return results_df, best_model_name, best_model, tuned_model, best_params


# 8 Prediction function
def predict_new_data(model, new_data_df, features):
    X_new = new_data_df[features]
    predictions = model.predict(X_new)
    return predictions


# 9 Load model function
def load_model(path="best_model.pkl"):
    return joblib.load(path)
