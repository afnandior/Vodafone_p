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

# Step1 Define function

def regression_pipeline_advanced(df, features, target, test_size=0.2, random_state=42, param_grids=None):
    """
    Advanced regression pipeline with:
    - Multiple models (Linear, Non-linear, Trees, Boosting, SVR, KNN, Neural Network)
    - Scaling
    - Cross-validation evaluation
    - Hyperparameter tuning if desired
    """

    print(" Starting Advanced Regression Pipeline...")

    # Step2 Data Preprocessing

    print(" Step 2: Preprocessing target column...")
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df = df.dropna(subset=[target] + features)



    # Step3 Split data
    
    print(" Step 3: Splitting data...")
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Step4 Define models
    print(" Step 4: Defining models...")
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

    # Step5 Define default param_grids if user didn't provide

    print(" Step 5: Setting hyperparameter grids...")
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

    # Step6 Train and evaluate models

    print(" Step 6: Training models...")
    results = []

    for name, model in models.items():
        print(f" Running {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

        print(f" {name} ➜ Test MSE: {mse:.2f} | Test R2: {r2:.4f} | CV R2: {cv_r2:.4f}")

        results.append({
            "Model": name,
            "Test_R2": r2,
            "CV_R2": cv_r2,
            "Test_MSE": mse,
            "Model_Object": model
        })


    # Step7 Sort results

    results_df = pd.DataFrame(results).sort_values(by="CV_R2", ascending=False)
    print("\n Step 7: Models sorted by CV R2:\n", results_df[["Model", "CV_R2", "Test_R2", "Test_MSE"]])


    # Step8 Select best model

    best_row = results_df.iloc[0]
    best_model_name = best_row["Model"]
    best_model = best_row["Model_Object"]

    print(f"\n Step 8: Best model is {best_model_name} ➜ CV R2 = {best_row['CV_R2']:.4f}")


    # Step9 Hyperparameter tuning if available

    print(" Step 9: Performing hyperparameter tuning if defined...")
    tuned_model = None

    if best_model_name in param_grids:
        grid = GridSearchCV(best_model, param_grids[best_model_name], cv=5, scoring="r2")
        grid.fit(X_train, y_train)
        tuned_model = grid.best_estimator_
        print(f" Best params for {best_model_name}: {grid.best_params_}")
    else:
        print(f" No hyperparameter tuning defined for {best_model_name}.")

    # Step10 Return results
  
    return results_df, best_model, tuned_model
