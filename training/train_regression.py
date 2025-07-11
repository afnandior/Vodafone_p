import pandas as pd
from modules.regression import regression_pipeline_advanced
import joblib

#  Load dataset
df = pd.read_csv("your_dataset.csv")

#  Specify features and target
features = ["feature1", "feature2"]   
target = "your_target_column"

#  Run training pipeline
results_df, best_model, tuned_model = regression_pipeline_advanced(df, features, target)

#  Save the tuned model if exists, else save the best model
final_model = tuned_model if tuned_model is not None else best_model
joblib.dump(final_model, "models/regression_model.pkl")

print("Training complete. Model saved to models/regression_model.pkl")
