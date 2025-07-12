# Import libraries
import pandas as pd
import numpy as np
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

# 1 Preprocessing Function
def preprocess_data(df, features, target):
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df = df.dropna(subset=[target] + features)
    X = df[features]
    y = df[target]
    return X, y


# 2 Data Split Function

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 3 Model Definitions Function

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

# 4 Train and Evaluate Models Function

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
        results.append({
            "Model": name,
            "Test_R2": r2,
            "CV_R2": cv_r2,
            "Test_MSE": mse,
            "Model_Object": model
        })
    results_df = pd.DataFrame(results).sort_values(by="CV_R2", ascending=False)
    return results_df

#  5 Hyperparameter Tuning Function

def tune_model(model, X_train, y_train, param_grid):
    grid = GridSearchCV(model, param_grid, cv=5, scoring="r2")
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


# 6 Main Regression Pipeline Function with default param_grids

def regression_pipeline_advanced(df, features, target, test_size=0.2, random_state=42, param_grids=None):
    X, y = preprocess_data(df, features, target)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    models = define_models()
    results_df = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)

    best_row = results_df.iloc[0]
    best_model_name = best_row["Model"]
    best_model = best_row["Model_Object"]

    #  Define default param_grids if not provided
    if param_grids is None:
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

    tuned_model = None
    best_params = None

    if best_model_name in param_grids:
        tuned_model, best_params = tune_model(best_model, X_train, y_train, param_grids[best_model_name])

    return results_df, best_model_name, best_model, tuned_model, best_params
