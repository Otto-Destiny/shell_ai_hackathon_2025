import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import time
import warnings
warnings.filterwarnings('ignore')


def calculate_cost(fractions_df: pd.DataFrame) -> np.ndarray:
    """Dummy cost function."""
    component_costs = np.array([10.5, 12.0, 9.8, 15.2, 11.5])
    return fractions_df.values @ component_costs


# --- IMPROVEMENT: A dedicated operator to handle the sum-to-one constraint ---
class NormalizationRepair(Repair):
    """
    A Repair operator that ensures the first 5 variables (fractions)
    in a solution always sum to 1. This is more efficient than
    treating it as a formal constraint during selection.
    """
    def _do(self, problem, X, **kwargs):
        # The input X is a matrix of solutions
        # Get the first 5 columns (the fractions)
        fractions = X[:, :5]

        # Calculate the sum of each row, preventing division by zero
        row_sums = np.sum(fractions, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 # Avoid division by zero

        # Normalize the fractions
        normalized_fractions = fractions / row_sums

        # Update the solutions in place
        X[:, :5] = normalized_fractions

        return X


class BlendOptimizationProblem(Problem):
    def __init__(self,
                 blend_model,
                 target_properties: np.ndarray,
                 fixed_inputs: np.ndarray,
                 frozen_targets: dict,
                 input_columns: list,
                 output_columns: list, # Now corresponds to the columns returned by predict_fast
                 optimize_cost=False):

        self.blend_model = blend_model
        self.target_properties = target_properties # Still need full targets to potentially select from
        self.fixed_inputs = fixed_inputs
        self.frozen_targets = frozen_targets
        self.input_columns = input_columns
        self.output_columns = output_columns # List of column names from predict_fast
        self.optimize_cost = optimize_cost
        self.is_empty = True if not frozen_targets else False

        # --- Constraint Handling ---
        # Constraints are only for frozen properties. Sum-to-one is handled by Repair.
        n_constraints = len(self.frozen_targets)

        n_obj = 2 if self.optimize_cost else 1

        super().__init__(
            n_var=5,
            n_obj=n_obj,
            n_constr=n_constraints if n_constraints > 0 else 1,
            xl=0.0,
            xu=1.0
        )

        self.epsilon = 1e-3

        # --- Adjusting objectives and constraints for predict_fast ---
        # The indices of the *predicted* properties that correspond to frozen targets
        self.frozen_indices_predicted = [self.output_columns.index(f'BlendProperty{k}') for k in self.frozen_targets.keys() if f'BlendProperty{k}' in self.output_columns]
        self.frozen_values = np.array([v for k, v in self.frozen_targets.items() if f'BlendProperty{k}' in self.output_columns])

        # The mask for the *predicted* properties that are NOT frozen (these are the objectives)
        self.objective_mask_predicted = np.ones(len(self.output_columns), dtype=bool)
        if self.frozen_indices_predicted:
             self.objective_mask_predicted[self.frozen_indices_predicted] = False

        # The target values for the properties that are objectives
        # Need to map the original target indices to the indices in the output_columns
        all_output_property_indices = [int(col.replace('BlendProperty', '')) for col in self.output_columns]
        self.objective_targets = np.array([
            self.target_properties[prop_idx - 1] # -1 because target_properties is 0-indexed
            for i, prop_idx in enumerate(all_output_property_indices)
            if self.objective_mask_predicted[i]
        ])


    def _evaluate(self, x, out, *args, **kwargs):
        # NOTE: The fractions 'x' are already normalized by the Repair operator,
        # so we can use them directly.

        # 1. Construct the full DataFrame for the model
        fixed_data = np.tile(self.fixed_inputs, (len(x), 1))
        full_input_data = np.hstack([x, fixed_data])
        input_df = pd.DataFrame(full_input_data, columns=self.input_columns)

        # 2. Get predictions (using predict_fast)
        predictions_df = self.blend_model.predict_fast(input_df)
        # Ensure predictions_df has the correct columns in the correct order
        predicted_properties = predictions_df[self.output_columns].values

        # 3. Calculate objectives
        # Use the mask specifically created for the predicted properties
        if self.is_empty:
            error = np.sum((predicted_properties[:, self.objective_mask_predicted] - self.objective_targets)**2, axis=1)
        else:
            error = np.sum((predicted_properties[:, self.frozen_indices_predicted] - self.frozen_values)**2, axis=1)
        if self.optimize_cost:
            cost = calculate_cost(input_df.iloc[:, :5])
            out["F"] = np.column_stack([error, cost])
        else:
            out["F"] = error

        # 4. Calculate constraints (ONLY for frozen properties predicted by predict_fast)
        if self.frozen_targets and self.frozen_indices_predicted:
             frozen_violations = np.abs(predicted_properties[:, self.frozen_indices_predicted] - self.frozen_values) - self.epsilon
             out["G"] = frozen_violations
        elif self.frozen_targets and not self.frozen_indices_predicted:
             # Frozen targets were provided but are not among the 'fast' properties.
             # This shouldn't happen with the current logic, but as a fallback,
             # return a dummy constraint violation indicating they can't be met via predict_fast.
             # A large positive number signifies a significant violation.
             out["G"] = np.full((len(x), len(self.frozen_targets)), 1e6) # Large violation for each specified frozen target
        else:
            # If there are no constraints, return a value of 0 (no violation).
            out["G"] = np.zeros(len(x))


# # input = np.array(train_df.iloc[1:1, :-10]).flatten() # np.random.rand(50) # Using dummy data as in Code A example
# # input_df = train_df.iloc[8:9, :-10]

# # --- 1. SETUP (same as before) ---
# blend_model = EagleBlendPredictor()
# fixed_model_inputs = np.array(train_df.iloc[5:6, 5:-10]).flatten() #np.random.rand(50)
# # input_cols = [f'frac_{i+1}' for i in range(5)] + [f'param_{i+1}' for i in range(50)]
# input_cols = train_df.iloc[:, :-10].columns.to_list()
# # Update output_cols to include ONLY the 6 properties returned by predict_fast
# output_cols = [f'BlendProperty{i}' for i in [1,2,5,6,7,10]]
# # Keep full_target_properties as it contains the values for all 10 properties
# full_target_properties = np.array([0.85, 1.2, 2.0, 1.25, 0.6, 0.45, 1.5, 0.80, 2.5, 1.0])


# # --- 2. CONFIGURE THE RUN ---

# # Use an empty dictionary to run without freezing any properties.
# # This should now work correctly and behave like your first script.
# frozen_targets_to_use = {1:1.7, 2:1.7}

# # Example with frozen targets (must be one of the 6 fast properties):
# # frozen_targets_to_use = {2: 0.5, 7: 0.1}

# INCLUDE_COST = True

# # --- 3. SETUP AND RUN ---
# problem = BlendOptimizationProblem(
#     blend_model=blend_model,
#     target_properties=full_target_properties, # Pass the full targets
#     fixed_inputs=fixed_model_inputs,
#     frozen_targets=frozen_targets_to_use,
#     input_columns=input_cols,
#     output_columns=output_cols, # Pass only the fast output column names
#     optimize_cost=INCLUDE_COST
# )

# # --- IMPROVEMENT: Add the Repair operator to the algorithm ---
# algorithm = NSGA2(
#     pop_size=100,
#     repair=NormalizationRepair(), # This automatically fixes the fractions
#     eliminate_duplicates=True
# )
# start = time.time()
# print("🚀 Starting Optimization (optimizing for fast properties only)...")
# res = minimize(
#     problem,
#     algorithm,
#     termination=('n_gen', 20),
#     seed=1,
#     verbose=True
# )
# end = time.time()
# print(f"Time taken: {end - start} seconds")

# # --- 4. ANALYZE RESULTS (same as before) ---
# print("\n--- ✅ Optimization Finished ---")
# if res.X is not None and len(res.X) > 0:
#     print(f"Found {len(res.X)} optimal trade-off solution(s).")
#     if INCLUDE_COST:
#         # For multi-objective, res.X contains the Pareto front of solutions
#         print(f"Found {len(res.X)} optimal trade-off solution(s).")
#         # Show the best solution in terms of property error (as in Code A)
#         best_by_error_idx = np.argmin(res.F[:, 0])
#         best_solution = res.X[best_by_error_idx]
#         best_objectives = res.F[best_by_error_idx]

#         normalized_sol = best_solution / np.sum(best_solution)
#         print("\n🏆 Best Solution (by smallest property error for fast properties):")
#         print(f"   Fractions: {np.round(normalized_sol, 4)}")
#         print(f"   Property Error (Objectives 1, 2, 5, 6, 7, 10): {best_objectives[0]:.4f}")
#         print(f"   Cost (Objective 2): {best_objectives[1]:.2f}")

#         # Plot the Pareto Front
#         plot = Scatter(title="Pareto Front: Error vs. Cost (Fast Properties)", labels=["Property Error (Fast)", "Cost"])
#         plot.add(res.F)
#         plot.show()

#     else:
#         # For single-objective, we have one best solution
#         print("Found best solution:")
#         normalized_sol = res.X / np.sum(res.X)
#         print(f"   Fractions: {np.round(normalized_sol, 3)}")
#         print(f"   Property Error (Objectives 1, 2, 5, 6, 7, 10): {res.F[0]:.4f}")

# else:
#     print("No feasible solution was found. If you are freezing properties, consider relaxing your constraints.")