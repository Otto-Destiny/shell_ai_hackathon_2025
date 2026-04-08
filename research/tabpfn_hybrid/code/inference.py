import pandas as pd
import numpy as np
import torch
import joblib
import argparse
import os
import glob
from sklearn.multioutput import MultiOutputRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
from tabpfn import TabPFNRegressor


os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "true"

def joblib_load_cpu(path):
	# Patch torch.load globally inside joblib to always load on CPU
	original_load = torch.load

	def cpu_loader(*args, **kwargs):
		kwargs['map_location'] = torch.device('cpu')
		return original_load(*args, **kwargs)

	torch.load = cpu_loader
	try:
		model = joblib.load(path)
	finally:
		torch.load = original_load  # Restore original torch.load
	return model

class TabPFNEnsemblePredictor:
	"""
	A class to load an ensemble of TabPFN models and generate averaged predictions.

	This class is designed to find and load all k-fold models from a specified
	directory, handle the necessary feature engineering, and produce a single,
	ensembled prediction from various input types (DataFrame, numpy array, or CSV file path).

	Attributes:
		model_paths (list): A list of file paths for the loaded models.
		models (list): A list of the loaded model objects.
		target_cols (list): The names of the target columns for the output DataFrame.
	"""
	
	def __init__(self, model_dir: str, model_pattern: str = "Fold_*_best_model.tabpfn_fit*"):
		"""
		Initializes the predictor by finding and loading the ensemble of models.

		Args:
			model_dir (str): The directory containing the saved .tabpfn_fit model files.
			model_pattern (str, optional): The glob pattern to find model files. 
										   Defaults to "Fold_*_best_model.tabpfn_fit".
		
		Raises:
			FileNotFoundError: If no models matching the pattern are found in the directory.
		"""
		print("Initializing the TabPFN Ensemble Predictor...")
		self.model_paths = sorted(glob.glob(os.path.join(model_dir, model_pattern)))
		if not self.model_paths:
			raise FileNotFoundError(
				f"Error: No models found in '{model_dir}' matching the pattern '{model_pattern}'"
			)
		
		print(f"Found {len(self.model_paths)} models to form the ensemble.")
		self.models = self._load_models()
		self.target_cols = [f"BlendProperty{i}" for i in range(1, 11)]

	def _load_models(self) -> list:
		"""
		Loads the TabPFN models from the specified paths and moves them to the CPU.
		
		This is a private method called during initialization.
		"""
		loaded_models = []
		for model_path in self.model_paths:
			print(f"Loading model: {os.path.basename(model_path)}...")
			try:
				# Move model components to CPU for inference to avoid potential CUDA errors
				# and ensure compatibility on machines without a GPU.
				if not torch.cuda.is_available():
					#torch.device("cpu")  # Force default
					#os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
					#os.environ["CUDA_VISIBLE_DEVICES"] = ""
					#os.environ["HSA_OVERRIDE_GFX_VERSION"] = "0"
					model = joblib_load_cpu(model_path)
					for estimator in model.estimators_:
						estimator.device = "cpu"
						estimator.max_time = 40
					print("Cuda not available using cpu")
					#for estimator in model.estimators_:
					#	if hasattr(estimator, "predictor_") and hasattr(estimator.predictor_, "predictors"):
					#		for p in estimator.predictor_.predictors:
					#			p.to("cpu")
					#	if hasattr(estimator.predictor_, 'to'):
					#		estimator.predictor_.to('cpu')

				else:
					print("Cuda is available")
					model = joblib.load(model_path)
					for estimator in model.estimators_:
						if hasattr(estimator, "predictor_") and hasattr(estimator.predictor_, "predictors"):
							for p in estimator.predictor_.predictors:
								p.to("cuda")
				
				loaded_models.append(model)
				print(f"Successfully loaded {os.path.basename(model_path)}")
			except Exception as e:
				print(f"Warning: Could not load model from {model_path}. Skipping. Error: {e}")
		return loaded_models

	@staticmethod
	def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
		"""
		Applies feature engineering to the input dataframe. This is a static method
		as it does not depend on the state of the class instance.

		Args:
			df (pd.DataFrame): The input dataframe.

		Returns:
			pd.DataFrame: The dataframe with new engineered features.
		"""
		components = ['Component1', 'Component2', 'Component3', 'Component4', 'Component5']
		properties = [f'Property{i}' for i in range(1, 11)]
		df_featured = df.copy()

		for prop in properties:
			df_featured[f'Weighted_{prop}'] = sum(
				df_featured[f'{comp}_fraction'] * df_featured[f'{comp}_{prop}'] for comp in components
			)
			cols = [f'{comp}_{prop}' for comp in components]
			df_featured[f'{prop}_variance'] = df_featured[cols].var(axis=1)
			df_featured[f'{prop}_range'] = df_featured[cols].max(axis=1) - df_featured[cols].min(axis=1)
			
		return df_featured

	def custom_predict(self, input_data: pd.DataFrame or np.ndarray or str) -> (np.ndarray, pd.DataFrame):
		"""
		Generates ensembled predictions for the given input data.

		This method takes input data, preprocesses it if necessary, generates a
		prediction from each model in the ensemble, and returns the averaged result.

		Args:
			input_data (pd.DataFrame or np.ndarray or str): The input data for prediction. 
				Can be a pandas DataFrame, a numpy array (must be pre-processed), 
				or a string path to a CSV file.

		Returns:
			tuple: A tuple containing:
				- np.ndarray: The averaged predictions as a numpy array.
				- pd.DataFrame: The averaged predictions as a pandas DataFrame.
		"""
		if not self.models:
			print("Error: No models were loaded. Cannot make predictions.")
			return None, None

		# --- Data Preparation ---
		if isinstance(input_data, str) and os.path.isfile(input_data):
			print(f"Loading and processing data from CSV: {input_data}")
			test_df = pd.read_csv(input_data)
			processed_df = self._feature_engineering(test_df)
		elif isinstance(input_data, pd.DataFrame):
			print("Processing input DataFrame...")
			processed_df = self._feature_engineering(input_data)
		elif isinstance(input_data, np.ndarray):
			print("Using input numpy array directly (assuming it's pre-processed).")
			sub = input_data
		else:
			raise TypeError("Input data must be a pandas DataFrame, a numpy array, or a path to a CSV file.")

		if isinstance(input_data, (str, pd.DataFrame)):
			if "ID" in processed_df.columns:
				sub = processed_df.drop(columns=["ID"]).values
			else:
				sub = processed_df.values
		
		# --- Prediction Loop ---
		all_fold_predictions = []
		print("\nGenerating predictions from the model ensemble...")
		for i, model in enumerate(self.models):
			try:
				y_sub = model.predict(sub)
				all_fold_predictions.append(y_sub)
				print(f"  - Prediction from model {i+1} completed.")
			except Exception as e:
				print(f"  - Warning: Could not predict with model {i+1}. Skipping. Error: {e}")
		
		if not all_fold_predictions:
			print("\nError: No predictions were generated from any model.")
			return None, None

		# --- Averaging ---
		print("\nAveraging predictions from all models...")
		averaged_preds_array = np.mean(all_fold_predictions, axis=0)
		averaged_preds_df = pd.DataFrame(averaged_preds_array, columns=self.target_cols)
		print("Ensemble prediction complete.")

		return averaged_preds_array, averaged_preds_df

# This block allows the script to be run directly from the command line
if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="""
		Command-line interface for the TabPFNEnsemblePredictor.
		
		Example Usage:
		python inference.py --model_dir ./saved_models/ --input_path ./test_data.csv --output_path ./final_preds.csv
		""",
		formatter_class=argparse.RawTextHelpFormatter
	)
	
	parser.add_argument("--model_dir", type=str, required=True, 
						help="Directory containing the saved .tabpfn_fit model files.")
	parser.add_argument("--input_path", type=str, required=True, 
						help="Path to the input CSV file for prediction.")
	parser.add_argument("--output_path", type=str, default="predictions_ensembled.csv",
						help="Path to save the final ensembled predictions CSV file.")

	args = parser.parse_args()

	if not os.path.isdir(args.model_dir):
		print(f"Error: Model directory not found at {args.model_dir}")
	elif not os.path.exists(args.input_path):
		print(f"Error: Input file not found at {args.input_path}")
	else:
		try:
			# 1. Instantiate the predictor class
			predictor = TabPFNEnsemblePredictor(model_dir=args.model_dir)

			# 2. Call the predict method
			preds_array, preds_df = predictor.predict(args.input_path)

			# 3. Save the results
			if preds_df is not None:
				preds_df.to_csv(args.output_path, index=False)
				print(f"\nEnsembled predictions successfully saved to {args.output_path}")
				print("\n--- Sample of Final Averaged Predictions ---")
				print(preds_df.head())
				print("------------------------------------------")

		except Exception as e:
			print(f"\nAn error occurred during the process: {e}")