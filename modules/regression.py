import pandas as pd
import numpy as np
import joblib
import datetime
import os
import json

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
import lightgbm as lgb


class RegressionTool:
    def __init__(self, save_dir="models"):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def remove_outliers_iqr(self, df, features):
        for col in features:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df

    def auto_preprocess(self, df, features, target):
        df[target] = pd.to_numeric(df[target], errors='coerce')
        df = df.dropna(subset=[target] + features)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        if numeric_cols:
            df = self.remove_outliers_iqr(df, numeric_cols + [target])

        scaler = RobustScaler()
        if numeric_cols:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        X = df[features]
        y = df[target]
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def define_models(self):
        return {
            "Linear Regression": Pipeline([("scaler", RobustScaler()), ("model", LinearRegression())]),
            "Ridge Regression": Pipeline([("scaler", RobustScaler()), ("model", Ridge())]),
            "Lasso Regression": Pipeline([("scaler", RobustScaler()), ("model", Lasso())]),
            "ElasticNet Regression": Pipeline([("scaler", RobustScaler()), ("model", ElasticNet())]),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": xgb.XGBRegressor(),
            "LightGBM": lgb.LGBMRegressor(),
            "Polynomial (deg=2)": Pipeline([("poly", PolynomialFeatures(degree=2)), ("scaler", RobustScaler()), ("linear", LinearRegression())]),
            "SVR": Pipeline([("scaler", RobustScaler()), ("model", SVR())]),
            "KNN Regression": Pipeline([("scaler", RobustScaler()), ("model", KNeighborsRegressor())]),
            "Neural Network": Pipeline([("scaler", RobustScaler()), ("model", MLPRegressor(max_iter=1000))]),
        }

    def define_param_grids(self):
        return {
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
            "Neural Network": {"model__hidden_layer_sizes": [(50,), (100,), (100, 50)], "model__activation": ["relu", "tanh"]},
        }

    def train(self, df, features, target, manual_hyperparams=None):
        df = df[features + [target]]
        X, y = self.auto_preprocess(df, features, target)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        models = self.define_models()
        param_grids = self.define_param_grids()

        results = []
        best_r2 = -np.inf
        best_model_name = None
        best_model = None
        best_params = None

        for name, model in models.items():
            try:
                if manual_hyperparams and name in manual_hyperparams:
                    model.set_params(**manual_hyperparams[name])

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

                this_best_params = manual_hyperparams.get(name) if manual_hyperparams else None

                if not (manual_hyperparams and name in manual_hyperparams) and name in param_grids:
                    grid = GridSearchCV(model, param_grids[name], cv=5, scoring="r2")
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                    this_best_params = grid.best_params_

                if cv_r2 > best_r2:
                    best_r2 = cv_r2
                    best_model_name = name
                    best_model = model
                    best_params = this_best_params

                results.append({
                    "Model": name,
                    "Test_R2": round(r2, 4),
                    "CV_R2": round(cv_r2, 4),
                    "Test_MSE": round(mse, 4),
                    "Best_Params": this_best_params
                })
            except Exception as e:
                results.append({"Model": name, "Error": str(e)})

        results_df = pd.DataFrame(results).sort_values(by="CV_R2", ascending=False).reset_index(drop=True)

        model_path = self.save_model_and_log(best_model, best_model_name)

        metadata = {
            "features": features,
            "target": target
        }
        with open(model_path.replace(".pkl", "_meta.json"), "w") as f:
            json.dump(metadata, f)

        return {
            "best_model": best_model_name,
            "best_params": best_params,
            "results": results_df,
            "model_path": model_path
        }

    def save_model_and_log(self, model, model_name):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name.replace(' ', '_')}_{timestamp}.pkl"
        model_path = os.path.join(self.save_dir, model_filename)
        joblib.dump(model, model_path)

        log_file = os.path.join(self.save_dir, "models_log.csv")
        log_entry = {
            "timestamp": timestamp,
            "model_name": model_name,
            "model_path": model_path
        }

        log_df = pd.DataFrame([log_entry])
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)

        return model_path

    def predict(self, model_path, new_data_df):
        model = joblib.load(model_path)

        metadata_path = model_path.replace(".pkl", "_meta.json")
        if not os.path.exists(metadata_path):
            raise ValueError("Metadata not found.")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        required_features = metadata["features"]
        missing = [f for f in required_features if f not in new_data_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        new_data_df = new_data_df[required_features]
        new_data_df = new_data_df.fillna(new_data_df.median(numeric_only=True))

        preds = model.predict(new_data_df)
        return preds.round(2)

    def predict_or_retrain(self, df, features, target, model_path=None):
        """
        لو الموديل موجود ومناسب، يستخدمه للتنبؤ.
        لو مش موجود أو الفيتشر مختلفة، يعمل تدريب جديد.
        """
        if model_path and os.path.exists(model_path):
            meta_path = model_path.replace(".pkl", "_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                if metadata["features"] == features and metadata["target"] == target:
                    return self.predict(model_path, df)
                else:
                    print(" Features or target don’t match saved model. Retraining a new model...")
            else:
                print(" Metadata not found. Retraining model...")
        else:
            print(" Model path not found. Training a new model...")

        result = self.train(df, features, target)
        return result["best_model"].predict(df[features])

    def read_models_log(self):
        log_file = os.path.join(self.save_dir, "models_log.csv")
        if os.path.exists(log_file):
            return pd.read_csv(log_file)
        else:
            return pd.DataFrame()
