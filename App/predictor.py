# prompt: import pandas and basic machine learning models for regression

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


from sklearn.model_selection import train_test_split

import itertools
import random

import torch
import random
import numpy as np

import os
import joblib

import matplotlib.pyplot as plt

from tabpfn import TabPFNRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_absolute_percentage_error

from sklearn.linear_model import LinearRegression

from inference import TabPFNEnsemblePredictor  # import inference.py

# from sklearn.metrics import mean_absolute_percentage_error
# from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
from itertools import combinations
from scipy.special import comb
# from tabpfn.model.loading import (
#     load_fitted_tabpfn_model,
#     save_fitted_tabpfn_model,
# )


class EagleBlendPredictor:
    def __init__(self, model_sources = './Models'):
        """
        model_sources: Dict[str, Any]
            A dictionary where keys are 'BlendProperty1', ..., 'BlendProperty10'
            and values are:
              - loaded model objects, or
              - callables returning models, or
              - custom loading logic (you will supply these)
        """
        self.home = model_sources
        self.saved_files_map = {
                      1: {
                          "model": 'linear_model_poly_target_1.joblib',
                          "transform": 'poly1_features.joblib'
                      },
                      2: {
                          "model": 'linear_model_poly_target_2.joblib',
                          "transform": 'poly2_features.joblib'
                      },
                      5: {
                          "model": 'tabpfn_model_target_5.joblib', #tabpfn_model_target_5_cpu.tabpfn_fit,'tabpfn_model_target_5_cpu.tabpfn_fit'
                          "transform": 'poly5_features.joblib'
                      },
                      6: {
                          "model": 'linear_model_poly_target_6.joblib',
                          "transform": 'poly6_features.joblib'
                      },
                      7: {
                          "model": 'tabpfn_model_target_7.joblib',
                          # For Property 7, the transformation is the mixture feature generation,
                          # which is not a saved object like PolynomialFeatures.
                          # You would need to apply the generate_mixture_features function.
                          "transform_function": "generate_mixture_features"
                      },
                      8: {
                          # For Property 8, the "model" is the initial prediction model (not explicitly saved in this workflow)
                          # and the correction is the piecewise function defined by parameters and threshold.
                          "params": 'piecewise_params_prop8.joblib',
                          "threshold": 'piecewise_threshold_prop8.joblib',
                          "correction_function": "piecewise_model" # Reference the function name
                      },
                      10: {
                          "model": 'linear_model_poly_target_10.joblib',
                          "transform": 'poly10_features.joblib'
                      }
                  }


        self.models = {}
        # Load models and transformers manually
        self.model_1 = joblib.load(os.path.join(self.home, self.saved_files_map[1]["model"]))
        self.poly_1 = joblib.load(os.path.join(self.home, self.saved_files_map[1]["transform"]))

        self.model_2 = joblib.load(os.path.join(self.home, self.saved_files_map[2]["model"]))
        self.poly_2 = joblib.load(os.path.join(self.home, self.saved_files_map[2]["transform"]))

        self.model_5 = joblib.load(
            os.path.join(self.home, self.saved_files_map[5]["model"]), #device="cpu"
        )
        self.poly_5 = joblib.load(os.path.join(self.home, self.saved_files_map[5]["transform"]))

        self.model_6 = joblib.load(os.path.join(self.home, self.saved_files_map[6]["model"]))
        self.poly_6 = joblib.load(os.path.join(self.home, self.saved_files_map[6]["transform"]))

        self.model_7 = joblib.load(
            os.path.join(self.home, self.saved_files_map[7]["model"]), #device="cpu"
        )
        # No saved transform for model_7 — use generate_mixture_features later in prediction
        self.piecewise_params_8 = joblib.load(os.path.join(self.home, self.saved_files_map[8]["params"]))
        self.piecewise_threshold_8 = joblib.load(os.path.join(self.home, self.saved_files_map[8]["threshold"]))

        # Use piecewise_model function later

        self.model_10 = joblib.load(os.path.join(self.home, self.saved_files_map[10]["model"]))
        self.poly_10 = joblib.load(os.path.join(self.home, self.saved_files_map[10]["transform"]))

        self.model_3489 = TabPFNEnsemblePredictor(model_dir=self.home)
        pass


    def piecewise_model(self, x, boundaries=np.linspace(-0.2, 0.2, 10+1)[1:-1]):
        """
        x: a single float value
        params: list of 20 parameters [A1, B1, A2, B2, ..., A10, B10]
        boundaries: 9 values that divide x into 10 regions
        """
        params = self.piecewise_params_8
        # Unpack parameters
        segments = [(params[i], params[i+1]) for i in range(0, 20, 2)]

        # Piecewise logic using boundaries
        for i, bound in enumerate(boundaries):
            if x < bound:
                A, B = segments[i]
                return A * x + B
        # If x is greater than all boundaries, use the last segment
        A, B = segments[-1]
        return A * x + B

    def predict_BlendProperty1(self, data, full = True):
        # Dummy custom transformation and prediction for BlendProperty1
        if full:
            features = self._transform1(data)
            features = self.poly_1.transform(features)
        else:
            features = self.poly_1.transform(data)
        res_df = self.model_1.predict(features)
        return pd.DataFrame(res_df, columns=['BlendProperty1'])


    def predict_BlendProperty2(self, data, full = True):
        if full:
            features = self._transform2(data)
            features = self.poly_2.transform(features)
        else:
            features = self.poly_2.transform(data)
        res_df = self.model_2.predict(features)
        return pd.DataFrame(res_df, columns=['BlendProperty2'])


    def predict_BlendProperty3489(self, df):
        arrray,result_df = self.model_3489.custom_predict(df)
        ans_df= result_df[['BlendProperty3','BlendProperty4','BlendProperty8','BlendProperty9']].copy() # Explicitly create a copy

        ans_df.loc[ans_df['BlendProperty8'].abs()<0.2,'BlendProperty8'] = ans_df[ans_df['BlendProperty8'].abs()<0.2]['BlendProperty8'].apply(self.piecewise_model)
        ans_df.loc[ans_df['BlendProperty9'].abs()<0.1,'BlendProperty9'] = 0 #ans_df[ans_df['BlendProperty8'].abs()<0.2]['BlendProperty8'].apply(self.piecewise_model)

        return ans_df

        # ndf.loc[ndf[pred_col].abs() < threshold_8, pred_col] = ndf[ndf[pred_col].abs() < threshold_8][pred_col].apply(func8)

    def predict_BlendProperty5(self, data, full =True ):
        if full:
            features = self._transform5(data)
            features = self.poly_5.transform(features)
        else:
            features = self.poly_5.transform(data)
        res_df = self.model_5.predict(features)
        return pd.DataFrame(res_df, columns=['BlendProperty5'])


    def predict_BlendProperty6(self, data, full=True):
        if full:
            features = self._transform6(data)
            features = self.poly_6.transform(features)
        else:
            features = self.poly_6.transform(data)
        res_df = self.model_6.predict(features)
        return pd.DataFrame(res_df, columns=['BlendProperty6'])


    def predict_BlendProperty7(self, data, full =True)-> pd.DataFrame:
        if full:
            features = self._transform7(data)
        else:
            raise ValueError("BlendProperty7 prediction requires full data.")
        res_df = self.model_7.predict(features)
        return pd.DataFrame(res_df, columns=['BlendProperty7'])


    def predict_BlendProperty10(self, data, full = False)-> pd.DataFrame:
        if full:
            features = self._transform10(data)
            features = self.poly_10.transform(features)
        else:
            features = self.poly_10.transform(data)
        res_df = self.model_10.predict(features)
        return pd.DataFrame(res_df, columns=['BlendProperty10'])


    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for all blend properties using the individual prediction methods.

        Args:
            df: Input DataFrame containing the features.

        Returns:
            DataFrame with predicted blend properties from 'BlendProperty1' to 'BlendProperty10'.
        """
        predictions_list = []

        # Predict individual properties
        predictions_list.append(self.predict_BlendProperty1(df, full=True))
        predictions_list.append(self.predict_BlendProperty2(df, full=True))

        # Predict BlendProperty3, 4, 8, and 9 together using predict_BlendProperty3489
        # Assuming predict_BlendProperty3489 returns a DataFrame with columns for these properties.
        predictions_3489_df = self.predict_BlendProperty3489(df)
        predictions_list.append(predictions_3489_df[['BlendProperty3']])
        predictions_list.append(predictions_3489_df[['BlendProperty4']])
        predictions_list.append(predictions_3489_df[['BlendProperty8']])
        predictions_list.append(predictions_3489_df[['BlendProperty9']])


        predictions_list.append(self.predict_BlendProperty5(df, full=True))
        predictions_list.append(self.predict_BlendProperty6(df, full=True))
        predictions_list.append(self.predict_BlendProperty7(df, full=True))


        predictions_list.append(self.predict_BlendProperty10(df, full=True))


        # Concatenate the list of single-column DataFrames into a single DataFrame
        predictions_df = pd.concat([df.reset_index(drop=True) for df in predictions_list], axis=1)


        # Ensure columns are in the desired order
        ordered_cols = [f'BlendProperty{i}' for i in range(1, 11)]
        # Reindex to ensure columns are in order, dropping any not generated (though all should be)
        predictions_df = predictions_df.reindex(columns=ordered_cols)


        return predictions_df






    # Dummy transformation functions (replace with your actual logic later)
    def _transform1(self, data):
        """
        Transforms input data (DataFrame or NumPy array) to features for BlendProperty1 prediction.

        If input is a DataFrame, selects 'ComponentX_fraction' (X=1-5) and 'ComponentX_Property1' (X=1-5).
        If input is a NumPy array, assumes the columns are already in the correct order:
        Component1-5_fraction, Component1-5_Property1, Component1-5_Property2, ..., Component1-5_Property10
        and selects the relevant columns for Property1.
        Args:
            data: pandas DataFrame or numpy array.

        Returns:
            numpy array of transformed features.
        """
        fraction_cols = [f'Component{i+1}_fraction' for i in range(5)]
        property_cols = [f'Component{i+1}_Property1' for i in range(5)]
        required_cols = fraction_cols + property_cols

        if isinstance(data, pd.DataFrame):
            # Select the required columns from the DataFrame
            # Ensure columns exist to avoid KeyError
            try:
                features = data[required_cols]
                print(features.columns)
            except KeyError as e:
                missing_col = str(e).split("'")[1]
                raise ValueError(f"Input DataFrame is missing required column: {missing_col}") from e

        elif isinstance(data, np.ndarray):
            # Assume the NumPy array has columns in the specified order
            # Select the first 5 columns (fractions) and columns for Property1 (indices 5 to 9)
            if data.shape[1] < 10: # Need at least 5 fractions and 5 properties
                raise ValueError(f"Input NumPy array must have at least 10 columns for this transformation.")

            # Selecting columns based on the assumed order: fractions (0-4), Property1 (5-9)
            features = data[:, :10] # Select first 10 columns: 5 fractions + 5 Property1

        else:
            raise TypeError("Input data must be a pandas DataFrame or a numpy array.")

        # Return as numpy array, as expected by PolynomialFeatures.transform
        return features

    def _transform2(self, data):
        """
        Transforms input data (DataFrame or NumPy array) to features for BlendProperty2 prediction.
        """
        fraction_cols = [f'Component{i+1}_fraction' for i in range(5)]
        property_cols = [f'Component{i+1}_Property2' for i in range(5)]
        required_cols = fraction_cols + property_cols

        if isinstance(data, pd.DataFrame):
            try:
                features = data[required_cols]
            except KeyError as e:
                missing_col = str(e).split("'")[1]
                raise ValueError(f"Input DataFrame is missing required column: {missing_col}") from e

        elif isinstance(data, np.ndarray):
            # Assume the NumPy array has columns in the specified order
            # Select the first 5 columns (fractions) and columns for Property2 (indices 10 to 14)
            if data.shape[1] < 15: # Need at least 5 fractions, 5 Property1, and 5 Property2
                raise ValueError(f"Input NumPy array must have at least 15 columns for this transformation.")

            # Selecting columns based on the assumed order: fractions (0-4), Property1 (5-9), Property2 (10-14)
            features = np.concatenate([data[:, :5], data[:, 10:15]], axis=1)

        else:
            raise TypeError("Input data must be a pandas DataFrame or a numpy array.")

        return features.values if isinstance(features, pd.DataFrame) else features

    def _transform3(self, data): return None

    def _transform4(self, data): return None

    def _transform5(self, data):
        """
        Transforms input data (DataFrame or NumPy array) to features for BlendProperty5 prediction.
        Args:
            data: pandas DataFrame or numpy array.

        Returns:
            numpy array of transformed features.
        """
        fraction_cols = [f'Component{i+1}_fraction' for i in range(5)]
        property_cols = [f'Component{i+1}_Property5' for i in range(5)]
        required_cols = fraction_cols + property_cols

        if isinstance(data, pd.DataFrame):
            try:
                features = data[required_cols]
            except KeyError as e:
                missing_col = str(e).split("'")[1]
                raise ValueError(f"Input DataFrame is missing required column: {missing_col}") from e

        elif isinstance(data, np.ndarray):
            # Assume the NumPy array has columns in the specified order
            # Select the first 5 columns (fractions) and columns for Property5 (indices 25 to 29)
            if data.shape[1] < 30: # Need at least 5 fractions and 5 properties for each of Property1-5
                raise ValueError(f"Input NumPy array must have at least 30 columns for this transformation.")

            # Selecting columns based on the assumed order: fractions (0-4), properties (5-9) for P1, (10-14) for P2, ..., (25-29) for P5
            features = np.concatenate([data[:, :5], data[:, 25:30]], axis=1)

        else:
            raise TypeError("Input data must be a pandas DataFrame or a numpy array.")

        return features


    def _transform6(self, data):
        """
        Transforms input data (DataFrame or NumPy array) to features for BlendProperty6 prediction.

        Args:
            data: pandas DataFrame or numpy array.

        Returns:
            numpy array of transformed features.
        """
        fraction_cols = [f'Component{i+1}_fraction' for i in range(5)]
        property_cols = [f'Component{i+1}_Property6' for i in range(5)]
        required_cols = fraction_cols + property_cols

        if isinstance(data, pd.DataFrame):
            try:
                features = data[required_cols]
            except KeyError as e:
                missing_col = str(e).split("'")[1]
                raise ValueError(f"Input DataFrame is missing required column: {missing_col}") from e

        elif isinstance(data, np.ndarray):
            # Assume the NumPy array has columns in the specified order
            # Select the first 5 columns (fractions) and columns for Property6 (indices 30 to 34)
            if data.shape[1] < 35: # Need at least 5 fractions and 5 properties for each of Property1-6
                raise ValueError(f"Input NumPy array must have at least 35 columns for this transformation.")

            # Selecting columns based on the assumed order: fractions (0-4), properties (5-9) for P1, ..., (30-34) for P6
            features = np.concatenate([data[:, :5], data[:, 30:35]], axis=1)

        else:
            raise TypeError("Input data must be a pandas DataFrame or a numpy array.")

        return features




    # def _transform7(self, row): return self._to_feature_array(row)
    # def _transform7(self, df: pd.DataFrame) -> pd.DataFrame:
    #     tn = 7
    #     fn = tn

    #     property_tn = [f'Component{i+1}_Property{fn}' for i in range(5)]
    #     fraction_cols = [f'Component{i+1}_fraction' for i in range(5)]
    #     target_cols = [f"BlendProperty{i}" for i in range(1, 11)]

    #     # Generate mixture features using the class method
    #     df_prop7 = df[fraction_cols + property_tn]
    #     mixture_features = self.generate_mixture_features(df_prop7)

    #     # Identify columns to concatenate (exclude targets and fraction/property 7 columns)
    #     exclude_cols = target_cols + fraction_cols + property_tn
    #     # other_features = df[[col for col in df.columns if col not in exclude_cols]]
    #     other_features = [f"Component{i}_Property{j}" for j in range(1,11) for i in range(1,6) if j!= 7]

    #     # Reset indices and concatenate
    #     mixture_features = mixture_features.reset_index(drop=True)
    #     # other_features = other_features.reset_index(drop=True)

    #     return pd.concat([mixture_features, df[other_features].copy()], axis=1)

    def _transform7(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrected transformation function for BlendProperty7 prediction.

        Args:
            df: Input DataFrame containing the features.

        Returns:
            DataFrame with generated features for BlendProperty7 prediction.
        """
        tn = 7
        fn = tn

        property_tn = [f'Component{i+1}_Property{fn}' for i in range(5)]
        fraction_cols = [f'Component{i+1}_fraction' for i in range(5)]

        # Generate mixture features
        df_prop7 = df[fraction_cols + property_tn].reset_index(drop=True) # Reset index here
        # Call the class's generate_mixture_features method
        mixture_features = self.generate_mixture_features(df_prop7)

        # Identify columns to concatenate (all ComponentX_PropertyY where Y != 7)
        other_property_cols = [f"Component{i}_Property{j}" for j in range(1,11) for i in range(1,6) if j!= 7]

        # Select these columns from the input DataFrame
        try:
            # Use .loc to preserve the original index when selecting columns, then reset index
            other_features_df = df.loc[:, other_property_cols].reset_index(drop=True) # Reset index here
        except KeyError as e:
            missing_col = str(e).split("'")[1]
            raise ValueError(f"Input DataFrame for _transform7 is missing required column: {missing_col}") from e


        # Concatenate along columns (axis=1). Indices should now be aligned after resetting.
        combined_features = pd.concat([mixture_features, other_features_df], axis=1)

        return combined_features

    def _transform8(self, row): return None
    def _transform9(self, row): return None

    def _transform10(self, data):
        """
        Transforms input data (DataFrame or NumPy array) to features for BlendProperty10 prediction.

        If input is a DataFrame, selects 'ComponentX_fraction' (X=1-5) and 'ComponentX_Property10' (X=1-5).
        If input is a NumPy array, assumes the columns are already in the correct order:
        Component1-5_fraction, Component1-5_Property1, Component1-5_Property2, ..., Component1-5_Property10
        and selects the relevant columns for Property10.

        Args:
            data: pandas DataFrame or numpy array.

        Returns:
            numpy array of transformed features.
        """
        fraction_cols = [f'Component{i+1}_fraction' for i in range(5)]
        property_cols = [f'Component{i+1}_Property10' for i in range(5)]
        required_cols = fraction_cols + property_cols

        if isinstance(data, pd.DataFrame):
            try:
                features = data[required_cols]
            except KeyError as e:
                missing_col = str(e).split("'")[1]
                raise ValueError(f"Input DataFrame is missing required column: {missing_col}") from e

        elif isinstance(data, np.ndarray):
            # Assume the NumPy array has columns in the specified order
            # Select the first 5 columns (fractions) and columns for Property10 (indices 50 to 54)
            if data.shape[1] < 55: # Need at least 5 fractions and 5 properties for each of Property1-10
                raise ValueError(f"Input NumPy array must have at least 55 columns for this transformation.")

            # Selecting columns based on the assumed order: fractions (0-4), properties (5-9) for P1, ..., (50-54) for P10
            features = np.concatenate([data[:, :5], data[:, 50:55]], axis=1)

        else:
            raise TypeError("Input data must be a pandas DataFrame or a numpy array.")

        return features



    def generate_mixture_features(self,data):
        """
        Generate symmetric and weighted nonlinear interactions between fuel weights and properties.
        The input 'data' should contain weights in the first 5 columns/elements and properties in the next 5.

        :param data: np.ndarray, pd.DataFrame, or list of shape (n_samples, 10) or (10,)
        :return: pd.DataFrame with generated features.
        """
        # Convert input to numpy array and handle single row/list input
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        elif isinstance(data, list):
            data_array = np.array(data)
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            raise TypeError("Input data must be a pandas DataFrame, numpy array, or list.")

        # Reshape single row/list input to 2D array
        if data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)

        # Ensure the input has 10 columns (5 weights + 5 properties)
        if data_array.shape[1] != 10:
            raise ValueError("Input data must have 10 columns/elements (5 weights and 5 properties).")

        # Separate weights and properties
        W = data_array[:, :5]
        P = data_array[:, 5:]


        n_samples, n_fuels = W.shape
        features = {}

        # Original weights and properties
        for i in range(n_fuels):
            features[f'w{i+1}'] = W[:, i]
            features[f'p{i+1}'] = P[:, i]
            features[f'w{i+1}_p{i+1}'] = W[:, i] * P[:, i]  # weighted property

        # --- 1. Weighted sum of properties ---
        features['weighted_sum'] = np.sum(W * P, axis=1)

        # --- 2. Weighted square of properties ---
        features['weighted_sum_sq'] = np.sum(W * P**2, axis=1)

        # --- 3. Weighted tanh of properties ---
        features['weighted_tanh'] = np.sum(W * np.tanh(P), axis=1)

        # --- 4. Weighted exponential ---
        # features['weighted_exp'] = np.sum(W * np.exp(P), axis=1)
        # Clip P before exponential to avoid overflow
        safe_exp = np.exp(np.clip(P, a_min=None, a_max=50))  # 50 is safe upper bound
        features['weighted_exp'] = np.sum(W * safe_exp, axis=1)


        # --- 5. Weighted logarithm (clip to avoid -inf) ---
        # features['weighted_log'] = np.sum(W * np.log(np.clip(P, 1e-6, None)), axis=1)
        features['weighted_log'] = np.sum(W * np.log(np.clip(P, 1e-6, None)), axis=1)


        # --- 6. Pairwise interactions (symmetric, weighted) ---
        for i, j in combinations(range(n_fuels), 2):
            pij = P[:, i] * P[:, j]
            wij = W[:, i] * W[:, j]
            features[f'pair_p{i+1}p{j+1}'] = pij
            features[f'weighted_pair_p{i+1}p{j+1}'] = pij * wij

        # --- 7. Triple interactions (weighted & symmetric) ---
        for i, j, k in combinations(range(n_fuels), 3):
            pij = P[:, i] * P[:, j] * P[:, k]
            wij = W[:, i] * W[:, j] * W[:, k]
            features[f'triplet_p{i+1}{j+1}{k+1}'] = pij
            features[f'weighted_triplet_p{i+1}{j+1}{k+1}'] = pij * wij

        # --- 8. Power series + weight modulated ---
        for power in [2, 3, 4]:
            features[f'power_sum_{power}'] = np.sum(W * P**power, axis=1)

        # --- 9. Log-weighted property (prevent log(0)) ---
        logW = np.log(np.clip(W, 1e-6, None))
        features['log_weighted_p'] = np.sum(logW * P, axis=1)

        # --- 10. Symmetric polynomial combinations (elementary symmetric) ---
        # Up to degree 5 (since you have 5 fuels)
        for r in range(1, 6):
            key = f'e_sym_poly_r{r}'
            val = np.zeros(n_samples)
            for idx in combinations(range(n_fuels), r):
                prod_p = np.prod(P[:, idx], axis=1)
                val += prod_p
            features[key] = val

        # --- 11. Weighted interaction difference (symmetry in differences) ---
        for i, j in combinations(range(n_fuels), 2):
            diff = P[:, i] - P[:, j]
            wdiff = W[:, i] * W[:, j]
            features[f'weighted_diff_p{i+1}{j+1}'] = diff * wdiff

        # --- 12. Mean, max, min (weighted) ---
        total_weight = np.sum(W, axis=1, keepdims=True)
        weighted_mean = np.sum(W * P, axis=1) / np.clip(total_weight.squeeze(), 1e-6, None)
        features['weighted_mean'] = weighted_mean
        features['max_prop'] = np.max(P, axis=1)
        features['min_prop'] = np.min(P, axis=1)

        # --- 13. Weighted cross-log terms ---
        for i, j in combinations(range(n_fuels), 2):
            log_mix = np.log(np.clip(P[:, i] + P[:, j], 1e-6, None))
            wij = W[:, i] * W[:, j]
            features[f'logsum_p{i+1}{j+1}'] = log_mix * wij

        # --- 14. Inverse + weighted inverse ---
        # features['inv_prop_sum'] = np.sum(W / np.clip(P, 1e-6, None), axis=1)
        features['inv_prop_sum'] = np.sum(W / np.clip(P, 1e-6, None), axis=1)


        # --- 15. Weighted relu (max(p, 0)) ---
        relu = np.maximum(P, 0)
        features['weighted_relu'] = np.sum(W * relu, axis=1)

        # --- 16. Weighted sin/cos transforms ---
        features['weighted_sin'] = np.sum(W * np.sin(P), axis=1)
        features['weighted_cos'] = np.sum(W * np.cos(P), axis=1)

        # --- 17. Normalized properties ---
        prop_sum = np.sum(P, axis=1, keepdims=True)
        normalized_P = P / np.clip(prop_sum, 1e-6, None)
        for i in range(n_fuels):
            features[f'norm_p{i+1}'] = normalized_P[:, i]

        # --- 18. Product of all p's and all w's ---
        features['total_product_p'] = np.prod(P, axis=1)
        features['total_product_w'] = np.prod(W, axis=1)

        # --- 19. Mixed entropic form ---
        # entropy_like = -np.sum(W * np.log(np.clip(W, 1e-6, None)), axis=1)
        # features['entropy_weights'] = entropy_like

        # Convert to DataFrame
        df = pd.DataFrame(features)

        return df
