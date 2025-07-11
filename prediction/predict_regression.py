import pandas as pd
import joblib

#  Load the saved model
model = joblib.load("models/regression_model.pkl")

#  Load new data for prediction
new_data = pd.read_csv("new_input.csv")   

#  Make predictions
predictions = model.predict(new_data)

#  Save or print predictions
output_df = pd.DataFrame(predictions, columns=["Predicted_Target"])
output_df.to_csv("data/predictions_output.csv", index=False)

print(" Prediction complete. Results saved to data/predictions_output.csv")
