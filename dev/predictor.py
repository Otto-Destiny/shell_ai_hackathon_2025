# inference.py
import joblib
import numpy as np
import os
import pandas as pd

class EagleBlendPredictor:
    def __init__(self, model_dir=r"C:\Users\Otto Henry\CodingWorld\Hackerthon\Fuel-Blend-Properties-Prediction\dev\models"):
        """
        Load fitted scaler, PCA transformer, and trained XGBoost multioutput model.
        """
        self.model_dir = model_dir
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        self.scaler = joblib.load(scaler_path)

        # Load PCA
        pca_path = os.path.join(model_dir, "pca.joblib") 
        self.pca = joblib.load(pca_path)

        # Load trained model
        model_path = os.path.join(model_dir, "xmodel.joblib")
        self.model = joblib.load(model_path)

    def predict_arr(self, X_new):
        """
        Make predictions on new data using the trained scaler, PCA, and model.
        
        Parameters:
        - X_new: array-like of shape (n_samples, n_features)
        
        Returns:
        - predictions: numpy array of shape (n_samples, n_outputs)
        """
        # Convert input to NumPy array
        X_new = np.array(X_new)

        # Step 1: Scale data
        X_scaled = self.scaler.transform(X_new)

        # Step 2: PCA transform
        X_pca = self.pca.transform(X_scaled)

        # Step 3: Predict
        predictions = self.model.predict(X_pca)

        return predictions

    def predict_all(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data using the trained scaler, PCA, and model.
        
        Parameters:
        - X_new: pandas DataFrame of shape (n_samples, n_features)
        
        Returns:
        - predictions_df: pandas DataFrame of shape (n_samples, 10) with blend property columns.
        """
        # Ensure the input is a DataFrame
        if not isinstance(X_new, pd.DataFrame):
            raise TypeError("Input X_new must be a pandas DataFrame.")

        # Step 1: Scale data
        # The scaler's transform method can directly handle a DataFrame
        # and will return a NumPy array.
        X_scaled = self.scaler.transform(X_new)

        # Step 2: PCA transform
        X_pca = self.pca.transform(X_scaled)

        # Step 3: Predict
        # The model's predict method returns a NumPy array.
        predictions_array = self.model.predict(X_pca)

        # Step 4: Format the output as a DataFrame
        # Create the desired column names
        column_names = [f'BlendProperty{i}' for i in range(1, 11)]
        
        # Create the DataFrame
        predictions_df = pd.DataFrame(predictions_array, columns=column_names, index=X_new.index)

        return predictions_df
    
    def predict_fast(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data using the trained scaler, PCA, and model.
        Only returns BlendProperty1, BlendProperty2, BlendProperty5, BlendProperty6,
        BlendProperty7, and BlendProperty10.
        
        Parameters:
        - X_new: pandas DataFrame of shape (n_samples, n_features)
        
        Returns:
        - predictions_df: pandas DataFrame of shape (n_samples, 6) 
                          with selected blend property columns.
        """
        if not isinstance(X_new, pd.DataFrame):
            raise TypeError("Input X_new must be a pandas DataFrame.")

        # Step 1: Scale data
        X_scaled = self.scaler.transform(X_new)

        # Step 2: PCA transform
        X_pca = self.pca.transform(X_scaled)

        # Step 3: Predict all blend properties
        predictions_array = self.model.predict(X_pca)

        # Step 4: Create DataFrame with all properties
        all_columns = [f'BlendProperty{i}' for i in range(1, 11)]
        predictions_df = pd.DataFrame(predictions_array, columns=all_columns, index=X_new.index)

        # Step 5: Select only required properties
        selected_columns = ['BlendProperty1', 'BlendProperty2', 
                            'BlendProperty5', 'BlendProperty6', 
                            'BlendProperty7', 'BlendProperty10']
        predictions_df = predictions_df[selected_columns]

        return predictions_df

# if __name__ == "__main__":
#     # Example usage
#     # Create the inference object
#     predictor = EagleBlendPredictor(model_dir="models")

#     # Example new data (must have same number of features as training data)
#     sample_data = [
#         [0.5, 1.2, 3.3, 4.1, 5.5],  # Replace with actual feature values
#         [1.5, 2.1, 0.3, 4.5, 2.5]
#     ]

#     # Get predictions
#     preds = predictor.predict_all(sample_data)
#     print("Predictions:\n", preds)
