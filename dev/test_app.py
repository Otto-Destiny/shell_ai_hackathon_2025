import streamlit as st

import pandas as pd
import matplotlib
import matplotlib.pyplot  as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import sqlite3
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import re
from pathlib import Path
import time # Add this import to the top of your script
import math
import plotly.graph_objects as go
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback


## -------------------------------------------------------------------------------------------------------------------------------------------
##                                                          Functions
##---------------------------------------------------------------------------------------------------------------------------------------------


class OptimizationProgressCallback(Callback):
    def __init__(self, progress_bar, total_generations):
        super().__init__()
        self.progress_bar = progress_bar
        self.total_generations = total_generations

    def notify(self, algorithm):
        # Calculate progress percentage
        progress_percent = algorithm.n_gen / self.total_generations
        
        # --- FIX: Changed the text to show percentage ---
        progress_text = f"Optimizing... {progress_percent:.0%}"
        
        # Update the Streamlit progress bar
        self.progress_bar.progress(progress_percent, text=progress_text)

class NormalizationRepair(Repair):
    """Ensures the first 5 variables (fractions) sum to 1."""
    def _do(self, problem, X, **kwargs):
        fractions = X[:, :5]
        row_sums = np.sum(fractions, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized_fractions = fractions / row_sums
        X[:, :5] = normalized_fractions
        return X
    
class BlendOptimizationProblem(Problem):
    def __init__(self, blend_model, target_properties, fixed_inputs, 
                 frozen_targets, input_columns, output_columns, optimize_cost=False):
        
        self.blend_model = blend_model
        self.target_properties = target_properties
        self.fixed_inputs = fixed_inputs
        self.frozen_targets = frozen_targets
        self.input_columns = input_columns
        self.output_columns = output_columns # Columns from predict_fast
        self.optimize_cost = optimize_cost
        
        n_constraints = len(self.frozen_targets)
        n_obj = 2 if self.optimize_cost else 1

        super().__init__(n_var=5, n_obj=n_obj, n_constr=n_constraints if n_constraints > 0 else 1, xl=0.0, xu=1.0)
        self.epsilon = 1e-3

        self.frozen_indices_predicted = [self.output_columns.index(f'BlendProperty{k}') for k in self.frozen_targets.keys() if f'BlendProperty{k}' in self.output_columns]
        self.frozen_values = np.array([v for k, v in self.frozen_targets.items() if f'BlendProperty{k}' in self.output_columns])
        
        self.objective_mask_predicted = np.ones(len(self.output_columns), dtype=bool)
        if self.frozen_indices_predicted:
            self.objective_mask_predicted[self.frozen_indices_predicted] = False
            
        all_output_prop_indices = [int(col.replace('BlendProperty', '')) for col in self.output_columns]
        self.objective_targets = np.array([
            self.target_properties[prop_idx - 1] 
            for i, prop_idx in enumerate(all_output_prop_indices) 
            if self.objective_mask_predicted[i]
        ])

    def _evaluate(self, x, out, *args, **kwargs):
        fixed_data = np.tile(self.fixed_inputs, (len(x), 1))
        full_input_data = np.hstack([x, fixed_data])
        input_df = pd.DataFrame(full_input_data, columns=self.input_columns)
        
        # STEP 1: Use predict_fast for optimization loop
        predicted_properties = self.blend_model.predict_fast(input_df)[self.output_columns].values
        
        error = np.sum((predicted_properties[:, self.objective_mask_predicted] - self.objective_targets)**2, axis=1)

        if self.optimize_cost:
            component_costs = np.array([st.session_state.get(f"opt_c{i}_cost", 0.0) for i in range(5)])
            cost = input_df.iloc[:, :5].values @ component_costs
            out["F"] = np.column_stack([error, cost])
        else:
            out["F"] = error

        if self.frozen_targets and self.frozen_indices_predicted:
            frozen_violations = np.abs(predicted_properties[:, self.frozen_indices_predicted] - self.frozen_values) - self.epsilon
            out["G"] = frozen_violations
        else:
            out["G"] = np.zeros(len(x))


# def run_real_optimization(targets, fixed_targets, components_data, include_cost):
#     """Main function to run the pymoo optimization."""
    
#     # 1. SETUP
#     blend_model = st.session_state.predictor
    
#     # All 55 input columns (5 fractions + 50 properties)
#     input_cols = [f'Component{i+1}_fraction' for i in range(5)]
#     for j in range(1, 11):
#         for i in range(1, 6):
#             input_cols.append(f'Component{i}_Property{j}')
            
#     # The 50 fixed property values from the UI
#     fixed_model_inputs = []
#     for j in range(1, 11):
#         for i in range(5):
#             fixed_model_inputs.append(st.session_state.get(f"opt_c{i}_prop{j}", 0.0))
#     fixed_model_inputs = np.array(fixed_model_inputs)

#     # STEP 2: Optimize based on only the 6 specified properties
#     output_cols_fast = [f'BlendProperty{i}' for i in [1, 2, 5, 6, 7, 10]]
    
#     full_target_properties = np.array(list(targets.values()))
#     frozen_targets_to_use = {int(k.replace('Property', '')): v for k, v in fixed_targets.items()}
    
#     # 2. RUN OPTIMIZATION
#     problem = BlendOptimizationProblem(
#         blend_model=blend_model, target_properties=full_target_properties,
#         fixed_inputs=fixed_model_inputs, frozen_targets=frozen_targets_to_use,
#         input_columns=input_cols, output_columns=output_cols_fast,
#         optimize_cost=include_cost
#     )
#     algorithm = NSGA2(pop_size=100, repair=NormalizationRepair(), eliminate_duplicates=True)
#     res = minimize(problem, algorithm, termination=('n_gen', 50), seed=1, verbose=False)
    
#     # 3. PROCESS AND RETURN RESULTS
#     if res.X is None or len(res.X) == 0:
#         st.error("Optimization failed to find a feasible solution. Consider relaxing your constraints.")
#         return []

#     # Prepare a full input DataFrame to get all 10 properties for the UI display
#     final_fractions_df = pd.DataFrame(res.X, columns=[f'Component{i+1}_fraction' for i in range(5)])
#     fixed_df_part = pd.DataFrame([fixed_model_inputs] * len(final_fractions_df), columns=input_cols[5:])
#     full_input_for_final_pred = pd.concat([final_fractions_df, fixed_df_part], axis=1)

#     # Use predict_all to get the full 10 properties for the UI, ensuring compatibility
#     all_10_properties_df = blend_model.predict_all(full_input_for_final_pred)
    
#     solutions = []
#     for i in range(len(res.X)):
#         solution_data = {
#             "component_fractions": res.X[i],
#             "blend_properties": all_10_properties_df.iloc[i].values, # Full 10 properties
#             "error": res.F[i][0],
#             "optimized_cost": res.F[i][1] if include_cost else 0.0
#         }
#         solutions.append(solution_data)
        
#     return solutions


def run_real_optimization(targets, fixed_targets, components_data, include_cost, generations, pop_size, progress_bar):
    """Main function to run the pymoo optimization."""

    # 1. SETUP (Remains the same)
    blend_model = st.session_state.predictor
    input_cols = [f'Component{i+1}_fraction' for i in range(5)]
    for j in range(1, 11):
        for i in range(1, 6):
            input_cols.append(f'Component{i}_Property{j}')

    fixed_model_inputs = []
    for j in range(1, 11):
        for i in range(5):
            fixed_model_inputs.append(st.session_state.get(f"opt_c{i}_prop{j}", 0.0))
    fixed_model_inputs = np.array(fixed_model_inputs)

    output_cols_fast = [f'BlendProperty{i}' for i in [1, 2, 5, 6, 7, 10]]

    full_target_properties = np.array(list(targets.values()))
    frozen_targets_to_use = {int(k.replace('Property', '')): v for k, v in fixed_targets.items()}

    # 2. RUN OPTIMIZATION (Remains the same)
    problem = BlendOptimizationProblem(
        blend_model=blend_model, target_properties=full_target_properties,
        fixed_inputs=fixed_model_inputs, frozen_targets=frozen_targets_to_use,
        input_columns=input_cols, output_columns=output_cols_fast,
        optimize_cost=include_cost
    )
    algorithm = NSGA2(pop_size=pop_size, repair=NormalizationRepair(), eliminate_duplicates=True)

    # Instantiate the callback with the progress bar and total generations
    callback = OptimizationProgressCallback(progress_bar, generations)
    
    # Add the 'callback' argument to the minimize function
    res = minimize(problem, algorithm, termination=('n_gen', generations), seed=1, verbose=False, callback=callback)
 

    # 3. PROCESS AND RETURN RESULTS (This section is modified)
    if res.X is None or len(res.X) == 0:
        st.error("Optimization failed to find a feasible solution. Consider relaxing your constraints.")
        return []

    # --- FIX: NO predict_all(). Instead, we build the final property list manually. ---

    # First, get the final *predicted* values for the 6 optimized properties
    final_fractions_df = pd.DataFrame(res.X, columns=[f'Component{i+1}_fraction' for i in range(5)])
    fixed_df_part = pd.DataFrame([fixed_model_inputs] * len(final_fractions_df), columns=input_cols[5:])
    full_input_for_fast_pred = pd.concat([final_fractions_df, fixed_df_part], axis=1)
    predicted_6_properties_df = blend_model.predict_fast(full_input_for_fast_pred)

    solutions = []
    optimized_prop_indices = [1, 2, 5, 6, 7, 10]

    for i in range(len(res.X)):
        # Create a 10-element array for the UI
        final_10_properties = np.zeros(10)

        for prop_idx in range(1, 11):
            if prop_idx in optimized_prop_indices:
                # For the 6 optimized properties, use the value from predict_fast
                col_name = f'BlendProperty{prop_idx}'
                final_10_properties[prop_idx - 1] = predicted_6_properties_df[col_name].iloc[i]
            else:
                # For the other 4, use the user's original target value as a placeholder
                final_10_properties[prop_idx - 1] = targets[f'Property{prop_idx}']

        solution_data = {
            "component_fractions": res.X[i],
            "blend_properties": final_10_properties, # Use the manually constructed array
            "error": res.F[i][0],
            "optimized_cost": res.F[i][1] if include_cost else 0.0
        }
        solutions.append(solution_data)
    solutions.sort(key=lambda x: x.get('error', float('inf')))
            
    return solutions


def calculate_quality_score(error, tolerance=1e-3):
    """Calculates a quality score, handling potential math errors."""
    if error is None or tolerance <= 0:
        return 0.0
    
    # If error is high, score is 0 to avoid math errors with log.
    if error >= tolerance:
        return 0.0
        
    # Prevent division by zero if error equals tolerance exactly in floating point.
    ratio = error / tolerance
    if ratio >= 1.0:
        return 0.0
        
    try:
        # The core formula
        score = 100 * (1 / (1 + math.log(1 - ratio)))
        # Ensure score is capped between 0 and 100
        return max(0, min(100, score))
    except (ValueError, TypeError):
        # Catch any other unexpected math errors
        return 0.0

@st.cache_data
def get_all_blends_data(db_path="eagleblend.db") -> pd.DataFrame:
    """Fetches all blend data, sorted by the most recent entries."""
    with sqlite3.connect(db_path) as conn:
        # Assuming 'id' is the primary key indicating recency
        query = "SELECT * FROM blends ORDER BY id DESC"
        df = pd.read_sql_query(query, conn)
    return df


def filter_component_options(df: pd.DataFrame, component_index: int) -> list:
    """
    Filters component options for a dropdown.
    - Primary filter: by 'component_type' matching the component index + 1.
    - Fallback filter: by 'component_name' ending with '_Component_{index+1}'.
    """
    target_type = component_index + 1

    # Primary Filter: Use 'component_type' if the column exists and has data.
    if 'component_type' in df.columns and not df['component_type'].isnull().all():
        # Use .loc to avoid SettingWithCopyWarning
        filtered_df = df.loc[df['component_type'] == target_type]
        if not filtered_df.empty:
            return filtered_df['component_name'].tolist()

    # Fallback Filter: If the primary filter fails or doesn't apply, use the name.
    # The 'na=False' gracefully handles any nulls in the component_name column.
    fallback_df = df.loc[df['component_name'].str.endswith(f"_Component_{target_type}", na=False)]
    return fallback_df['component_name'].tolist()


# ---------------------- Page Config ----------------------
st.set_page_config(
    layout="wide",
    page_title="Eagle Blend Optimizer",
    page_icon="🦅",
    initial_sidebar_state="collapsed"
)


# ---------------------- Sidebar Content ----------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🦅 Developed by eagle-team")
    st.markdown("""
    - Destiny Otto
    - Williams Alabi
    - Godswill Otto
    - Alexander Ifenaike
    """)
    st.markdown("---")
    st.info("Select a tab above to get started.")

# ---------------------- Custom Styling ---------------------- ##e0e0e0;

st.markdown("""
    <style>
            
    .block-container {
        padding-top: 1rem;
    }
    /* Main app background */
    .stApp {
        background-color: #f8f5f0;
        overflow: visible;
        padding-top: 0

        /* --- ADD THIS CSS FOR THE NEW HELP BUTTONS --- */
    #help-toggle-insights:checked ~ .help-panel-insights,
    #help-toggle-registry:checked ~ .help-panel-registry,
    #help-toggle-comparison:checked ~ .help-panel-comparison {
        opacity: 1; visibility: visible; transform: translateY(0);
    }
                 
    }
        /* Remove unnecessary space at the top */     
   /* Remove any fixed headers */
    .stApp > header {
        position: static !important;
    }           
    
    /* Header styling */
    .header {
        background: linear-gradient(135deg, #654321 0%, #8B4513 100%);
        color: white;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric card styling */
    .metric-card {
        background: #ffffff;  /* Pure white cards for contrast */
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        height: 100%;
        transition: all 0.3s ease;
        border: 1px solid #CFB53B;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        background: #FFF8E1;  /* Very light blue tint on hover */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border-color: #8B4513;
    }
    
    /* Metric value styling */
    .metric-value {
        color: #8B4513 !important;  /* Deep, vibrant blue */
        font-weight: 700;
        font-size: 1.8rem;
        text-shadow: 0 1px 2px rgba(0, 82, 204, 0.1);
    }
    
    /* Metric label styling */
    .metric-label {
        color: #654321;  /* Navy blue-gray */
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    
    /* Metric delta styling */
    .metric-delta {
        color: #A67C52;  /* Medium blue-gray */
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Tab styling */
    /* Main tab container */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 6px;
        padding: 8px;
        margin: 0 auto;
        width: 95% !important;
    }
    
    /* Individual tabs */
    .stTabs [data-baseweb="tab"] {
        flex: 1;  /* Equal width distribution */
        min-width: 0;  /* Allows flex to work */
        height: 60px;  /* Fixed height or use aspect ratio */
        padding: 0 12px;
        margin: 0;
        font-weight: 600;
        font-size: 1rem;
        color: #654321;
        background: #FFF8E1;
        border: 2px solid #CFB53B;
        border-radius: 12px;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    
    /* Hover state */
    .stTabs [data-baseweb="tab"]:hover {
        background: #FFE8A1;
        transform: translateY(-2px);
    }
    
    
    /* Active tab */
    .stTabs [aria-selected="true"] {
        background: #654321;
        color: #FFD700 !important;
        border-color: #8B4513;
        font-size: 1.05rem;
    }
    
    /* Icon sizing */
    .stTabs [data-baseweb="tab"] svg {
        width: 24px !important;
        height: 24px !important;
        margin-right: 8px !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #654321;
        color: #FFD700 !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #8B4513;
        color: white;
    }
    
    /* Dataframe styling */
    .table-container {
            display: flex;
            justify-content: center;
            margin-top: 30px;
    }
    .table-inner {
            width: 50%;
    }


    @media only screen and (max-width: 768px) {
        .table-inner {
            width: 90%; /* For mobile */
        }
    }
            
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        background-color:white !important;
        border: #CFB53B !important;
    }
    

    
    /* Section headers */
    .st-emotion-cache-16txtl3 {
        padding-top: 1rem;
    }
    
    /* Custom hr style */
    .custom-divider {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #dee2e6, transparent);
        margin: 2rem 0;
    }
            
    /* Consistent chart styling --- THIS IS THE FIX --- */
    .stPlotlyChart {
        border-radius: 10px;

        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
    }

    
    /* Match number inputs */
    # .stNumberInput > div {
    #     padding: 0.25rem 0.5rem !important;
    # }

    #/* Better select widget alignment */
    # .stSelectbox > div {
    #     margin-bottom: -15px;
    # }
            
    
    .custom-uploader > label div[data-testid="stFileUploadDropzone"] {
        border: 2px solid #4CAF50;
        background-color: #4CAF50;
        color: white;
        padding: 0.6em 1em;
        border-radius: 0.5em;
        text-align: center;
        cursor: pointer;
    }
    .custom-uploader > label div[data-testid="stFileUploadDropzone"]:hover {
        background-color: #45a049;
    }

    
    /* --- Add this CSS class for the spinner --- */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .spinner {
        border: 4px solid rgba(0,0,0,0.1);
        border-left-color: #8B4513;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        animation: spin 1s linear infinite;
    }
            
    
    /* Color scale adjustments */
    .plotly .colorbar {
        padding: 10px !important;
            color: #654321 !important;
    }   

    </style>
""", unsafe_allow_html=True)

# ---------------------- App Header ----------------------
# --- This is the new header with the subtitle ---
st.markdown("""
    <div class="header">
        <h1 style='text-align: center; margin-bottom: 0.5rem;'>🦅 Eagle Blend Optimizer</h1>
        <h4 style='text-align: center; font-weight: 400; margin-top: 0;'>
            AI-Powered Fuel Blend Property Prediction & Optimization
        </h4>
        <p style='text-align: center; font-weight: 300; font-size: 1rem; margin-top: 0.75rem; opacity: 0.9;'>
            by <b>eagle-team</b> for the Shell.ai 2025 Hackathon
        </p>
    </div>
""", unsafe_allow_html=True)
#------ universal variables
 

# ---------------------- Tabs ----------------------
tabs = st.tabs([
    "📊 Dashboard",
    "🎛️ Blend Designer",
    "⚙️ Optimization Engine",
    "📤 Blend Comparison",
    "📚 Fuel Registry",
    "🧠 Model Insights"
])


def explode_blends_to_components(blends_df: pd.DataFrame,
                                 n_components: int = 5,
                                 keep_empty: bool = False,
                                 blend_name_col: str = "blend_name") -> pd.DataFrame:
    """
    Convert a blends DataFrame into a components DataFrame.

    Parameters
    ----------
    blends_df : pd.DataFrame
        DataFrame with columns following the pattern:
        Component1_fraction, Component1_Property1..Property10, Component1_unit_cost, ...
    n_components : int
        Number of components per blend (default 5).
    blend_name_col : str
        Column name in blends_df that stores the blend name.

    Returns
    -------
    pd.DataFrame
        components_df with columns:
        ['blend_name', 'component_name', 'component_fraction',
         'property1', ..., 'property10', 'unit_cost']
    """

    components_rows = []
    prop_names = [f"property{i}" for i in range(1, 11)]

    for _, blend_row in blends_df.iterrows():
        blend_name = blend_row.get(blend_name_col)
        # Fallback if blend_name is missing/empty - keep index-based fallback
        if not blend_name or str(blend_name).strip() == "":
            # use the dataframe index + 1 to create a fallback name
            blend_name = f"blend{int(blend_row.name) + 1}"

        for i in range(1, n_components + 1):
            # Build column keys
            frac_col = f"Component{i}_fraction"
            unit_cost_col = f"Component{i}_unit_cost"
            prop_cols = [f"Component{i}_Property{j}" for j in range(1, 11)]

            # Safely get values (if column missing, get NaN)
            comp_frac = blend_row.get(frac_col, np.nan)
            comp_unit_cost = blend_row.get(unit_cost_col, np.nan)
            comp_props = [blend_row.get(pc, np.nan) for pc in prop_cols]

            row = {
                "blend_name": blend_name,
                "component_name": f"{blend_name}_Component_{i}",
                "component_fraction": comp_frac,
                "component_type": i,
                "unit_cost": comp_unit_cost
            }
            # add property1..property10
            for j, v in enumerate(comp_props, start=1):
                row[f"property{j}"] = v

            components_rows.append(row)

    components_df = pd.DataFrame(components_rows)

    return components_df

# --- Updated add_blends (now also populates components) ---
def add_blends(df, db_path="eagleblend.db", n_components=5):
    df = df.copy()

    # 1) Ensure blend_name column
    for col in list(df.columns):
        low = col.strip().lower()
        if low in ("blend_name", "blend name", "blendname"):
            if col != "blend_name":
                df = df.rename(columns={col: "blend_name"})
            break
    if "blend_name" not in df.columns:
        df["blend_name"] = pd.NA

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 2) Determine next blend number
    cur.execute("SELECT blend_name FROM blends WHERE blend_name LIKE 'blend%'")
    nums = [int(m.group(1)) for (b,) in cur.fetchall() if (m := re.match(r"blend(\d+)$", str(b)))]
    start_num = max(nums) if nums else 0

    # 3) Fill missing blend_name
    mask = df["blend_name"].isna() | (df["blend_name"].astype(str).str.strip() == "")
    df.loc[mask, "blend_name"] = [f"blend{i}" for i in range(start_num + 1, start_num + 1 + mask.sum())]

    # 4) Safe insert into blends
    cur.execute("PRAGMA table_info(blends)")
    db_cols = [r[1] for r in cur.fetchall()]
    safe_df = df[[c for c in df.columns if c in db_cols]]
    if not safe_df.empty:
        safe_df.to_sql("blends", conn, if_exists="append", index=False)

    # 5) Explode blends into components and insert into components table
    components_df = explode_blends_to_components(df, n_components=n_components, keep_empty=False)
    cur.execute("PRAGMA table_info(components)")
    comp_cols = [r[1] for r in cur.fetchall()]
    safe_components_df = components_df[[c for c in components_df.columns if c in comp_cols]]
    if not safe_components_df.empty:
        safe_components_df.to_sql("components", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()

    return {
        "blends_inserted": int(safe_df.shape[0]),
        "components_inserted": int(safe_components_df.shape[0])
    }


# --- add_components function ---
def add_components(df, db_path="eagleblend.db"):
    df = df.copy()

    # Ensure blend_name exists
    for col in list(df.columns):
        low = col.strip().lower()
        if low in ("blend_name", "blend name", "blendname"):
            if col != "blend_name":
                df = df.rename(columns={col: "blend_name"})
            break
    if "blend_name" not in df.columns:
        df["blend_name"] = pd.NA

    # Ensure component_name exists
    if "component_name" not in df.columns:
        df["component_name"] = pd.NA

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Fill missing component_name
    mask = df["component_name"].isna() | (df["component_name"].astype(str).str.strip() == "")
    df.loc[mask, "component_name"] = [
        f"{bn}_Component_{i+1}"
        for i, bn in enumerate(df["blend_name"].fillna("blend_unknown"))
    ]

    # Safe insert into components
    cur.execute("PRAGMA table_info(components)")
    db_cols = [r[1] for r in cur.fetchall()]
    safe_df = df[[c for c in df.columns if c in db_cols]]
    if not safe_df.empty:
        safe_df.to_sql("components", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()

    return int(safe_df.shape[0])

def get_blends_overview(db_path: str = "eagleblend.db", last_n: int = 5) -> Dict[str, Any]:
    """
    Returns:
      {
        "max_saving": float | None,          # raw numeric (PreOpt_Cost - Optimized_Cost)
        "last_blends": pandas.DataFrame,     # last_n rows of selected columns
        "daily_counts": pandas.Series        # counts per day, index = 'YYYY-MM-DD' (strings)
      }
    """
    last_n = int(last_n)
    comp_cols = [
        "blend_name", "Component1_fraction", "Component2_fraction", "Component3_fraction",
        "Component4_fraction", "Component5_fraction", "created_at"
    ]
    blend_props = [f"BlendProperty{i}" for i in range(1, 11)]
    select_cols = comp_cols + blend_props
    cols_sql = ", ".join(select_cols)

    with sqlite3.connect(db_path) as conn:
        # 1) scalar: max saving
        max_saving = conn.execute(
            "SELECT MAX(PreOpt_Cost - Optimized_Cost) "
            "FROM blends "
            "WHERE PreOpt_Cost IS NOT NULL AND Optimized_Cost IS NOT NULL"
        ).fetchone()[0]

        # 2) last N rows (only selected columns)
        q_last = f"""
            SELECT {cols_sql}
            FROM blends
            ORDER BY id DESC
            LIMIT {last_n}
        """
        df_last = pd.read_sql_query(q_last, conn)

        # 3) daily counts (group by date)
        q_counts = """
            SELECT date(created_at) AS day, COUNT(*) AS cnt
            FROM blends
            WHERE created_at IS NOT NULL
            GROUP BY day
            ORDER BY day DESC
        """
        df_counts = pd.read_sql_query(q_counts, conn)

    # Convert counts to a Series with day strings as index (fast, small memory)
    if not df_counts.empty:
        daily_counts = pd.Series(df_counts["cnt"].values, index=df_counts["day"].astype(str))
        daily_counts.index.name = "day"
        daily_counts.name = "count"
    else:
        daily_counts = pd.Series(dtype=int, name="count")

    return {"max_saving": max_saving, "last_blends": df_last, "daily_counts": daily_counts}


def get_activity_logs(db_path="eagleblend.db", timeframe="today", activity_type=None):
    """
    Get counts of activities from the activity_log table within a specified timeframe.

    Args:
        db_path (str): Path to the SQLite database file.
        timeframe (str): Time period to filter ('today', 'this_week', 'this_month', or 'custom').
        activity_type (str): Specific activity type to return count for. If None, return all counts.
    
    Returns:
        dict: Dictionary with counts per activity type OR a single integer if activity_type is specified.
    """
    # Calculate time filter
    now = datetime.now()
    if timeframe == "today":
        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif timeframe == "this_week":
        start_time = now - timedelta(days=now.weekday())  # Monday of this week
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    elif timeframe == "this_month":
        start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError("Invalid timeframe. Use 'today', 'this_week', or 'this_month'.")

    # Query database
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT activity_type, COUNT(*) as count
        FROM activity_log
        WHERE timestamp >= ?
        GROUP BY activity_type
    """
    df_counts = pd.read_sql_query(query, conn, params=(start_time.strftime("%Y-%m-%d %H:%M:%S"),))
    conn.close()

    # Convert to dictionary
    counts_dict = dict(zip(df_counts["activity_type"], df_counts["count"]))

    # If specific activity requested
    if activity_type:
        return counts_dict.get(activity_type, 0)
    
    return counts_dict

# print(get_activity_logs(timeframe="today"))          # All activities today
# print(get_activity_logs(timeframe="this_week"))      # All activities this week
# print(get_activity_logs(timeframe="today", activity_type="optimization"))  # Only optimization count today

# result = get_activity_logs(timeframe="this_week")
# result['optimization']
# result['prediction']


def get_model(db_path="eagleblend.db"):
    """
    Fetch the last model from the models_registry table.
    
    Returns:
        pandas.Series: A single row containing the last model's data.
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM models_registry ORDER BY id DESC LIMIT 1"
    df_last = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df_last.empty:
        return df_last.iloc[0]  # Return as a Series so you can access columns easily
    else:
        return None


# last_model = get_model()
# if last_model is not None:
#     print("R2 Score:", last_model["R2_Score"])


# ----------------------------------------------------------------------------------------------------------------------------------------------
#                                                       Dashboard Tab
# ----------------------------------------------------------------------------------------------------------------------------------------------
with tabs[0]:

    # NOTE: Assuming these functions are defined elsewhere in your application
    # from your_utils import get_model, get_activity_logs, get_blends_overview

    # ---------- formatting helpers ----------
    def fmt_int(x):
        try:
            return f"{int(x):,}"
        except Exception:
            return "0"

    def fmt_pct_from_r2(r2):
        if r2 is None:
            return "—"
        try:
            v = float(r2)
            if v <= 1.5:
                v *= 100.0
            return f"{v:.1f}%"
        except Exception:
            return "—"

    def fmt_currency(x):
        try:
            return f"${float(x):,.2f}"
        except Exception:
            return "—"

    # ---------- pull live data (this_week only) ----------
    # This block is assumed to be correct and functional
    try:
        last_model = get_model()
    except Exception as e:
        last_model = None
        st.warning(f"Model lookup failed: {e}")

    try:
        activity_counts = get_activity_logs(timeframe="this_week")
    except Exception as e:
        activity_counts = {}
        st.warning(f"Activity log lookup failed: {e}")

    try:
        overview = get_blends_overview(last_n=5)
    except Exception as e:
        overview = {"max_saving": None, "last_blends": pd.DataFrame(), "daily_counts": pd.Series(dtype=int)}
        st.warning(f"Blends overview failed: {e}")


    r2_display = fmt_pct_from_r2(None if last_model is None else last_model.get("R2_Score"))
    preds = fmt_int(activity_counts.get("prediction", 0))
    opts = fmt_int(activity_counts.get("optimization", 0))
    max_saving_display = fmt_currency(overview.get("max_saving", None))

    # ---------- KPI cards ----------
    # FIXED: Replaced st.subheader with styled markdown for consistent color
    st.markdown('<h2 style="color:#4a2f1f; font-size:1.75rem;">Performance Summary</h2>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
            <div class="metric-card" style="padding:10px;">
                <div class="metric-label">Model Accuracy</div>
                <div class="metric-value" style="font-size:1.3rem;">{r2_display}</div>
                <div class="metric-delta">R² (latest)</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
            <div class="metric-card" style="padding:10px;">
                <div class="metric-label">Predictions Made</div>
                <div class="metric-value" style="font-size:1.3rem;">{preds}</div>
                <div class="metric-delta">This Week</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
            <div class="metric-card" style="padding:10px;">
                <div class="metric-label">Optimizations</div>
                <div class="metric-value" style="font-size:1.3rem;">{opts}</div>
                <div class="metric-delta">This Week</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
            <div class="metric-card" style="padding:10px;">
                <div class="metric-label">Highest Cost Savings</div>
                <div class="metric-value" style="font-size:1.3rem;">{max_saving_display}</div>
                <div class="metric-delta">Per unit fuel</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    # ---------- Floating "How to Use" (bigger button + inline content) + compact CSS ----------
    # st.markdown("""
    # <style>
    # /* Floating help - larger button and panel */
    # #help-toggle{display:none;}
    # .help-button{
    #     position:fixed; right:25px; bottom:25px; z-index:9999;
    #     background:#8B4513; color:#FFD700; padding:16px 22px; font-size:17px;
    #     border-radius:18px; font-weight:900; box-shadow:0 8px 22px rgba(0,0,0,0.2); cursor:pointer;
    #     border:0;
    # }
    # .help-panel{
    #     position:fixed; right:25px; bottom:100px; z-index:9998;
    #     width:520px; max-height:70vh; overflow-y:auto;
    #     background: linear-gradient(135deg, #FFFDF5 0%, #F8EAD9 100%);
    #     border:1px solid #CFB53B; border-radius:12px; padding:20px; box-shadow:0 14px 34px rgba(0,0,0,0.22);
    #     color:#4a2f1f; transform: translateY(12px); opacity:0; visibility:hidden; transition: all .22s ease-in-out;
    # }
    # #help-toggle:checked + label.help-button + .help-panel{
    #     opacity:1; visibility:visible; transform: translateY(0);
    # }
    # .help-panel .head{display:flex; justify-content:space-between; align-items:center; margin-bottom:12px}
    # .help-panel .title{font-weight:900; color:#654321; font-size:16px}
    # .help-close{background:#8B4513; color:#FFD700; padding:6px 10px; border-radius:8px; cursor:pointer; font-weight:800}
    # .help-body{font-size:14.5px; color:#4a2f1f; line-height:1.5}
    # .help-body b {color: #654321;}

    # /* compact recent blends styles - improved font sizes */
    # .recent-compact { padding-left:6px; padding-right:6px; }
    # .compact-card{
    #     background: linear-gradient(180deg,#FFF8E1 0%, #FFF6EA 100%);
    #     border:1px solid #E3C77A; border-radius:8px; padding:10px; margin-bottom:8px; color:#654321;
    #     box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    # }
    # .compact-top{display:flex; justify-content:space-between; align-items:center; margin-bottom:8px}
    # .compact-name{font-weight:800; font-size:15px}
    # .compact-ts{font-size:12px; color:#8B4513; opacity:0.95; font-weight:700}
    # .comp-pills{font-size:12.5px; margin-bottom:8px}
    # .comp-pill{
    #     display:inline-block; padding:3px 8px; margin-right:6px; margin-bottom: 4px; border-radius:999px;
    #     background:rgba(139,69,19,0.06); border:1px solid rgba(139,69,19,0.12);
    #     font-weight:700; color:#654321;
    # }
    # .props-inline{
    #     font-size:12px; color:#4a2f1f; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    # }
    # .props-inline small{ font-size:11px; color:#4a2f1f; opacity:0.95; margin-right:8px; }
    # </style>

    # <input id="help-toggle" type="checkbox" />
    # <label for="help-toggle" class="help-button">💬 Help</label>

    # <div class="help-panel" aria-hidden="true">
    #     <div class="head">
    #         <div class="title">How to Use the Optimizer</div>
    #         <label for="help-toggle" class="help-close">Close</label>
    #     </div>
    #     <div class="help-body">
    #         <p><b>Performance Cards:</b> These show key metrics at a glance. "Model Accuracy" is the latest R² score. "Predictions" and "Optimizations" cover this week's activity. If a card shows "—", the underlying data may be missing.</p>
    #         <p><b>Blend Entries Chart:</b> This chart tracks how many new blends are created each day. Spikes can mean heavy usage or batch imports, while gaps might point to data ingestion issues.</p>
    #         <p><b>Recent Blends:</b> This is a live list of the newest blends. Each card displays the blend's name, creation time, component mix (C1-C5), and key properties (P1-P10). You can use the name and timestamp to find the full record in the database.</p>
    #         <p><b>Operational Tips:</b> For best results, use consistent naming for your blends. Ensure your data includes cost fields for savings to be calculated correctly. Consider retraining your model if its accuracy drops.</p>
    #     </div>
    # </div>
    # """, unsafe_allow_html=True)

# --- FIX: Removed extra blank lines inside the <ul> tag to ensure all items render ---
    st.markdown("""
        <style>
        /* Floating help - larger button and panel */
        #help-toggle{display:none;}
        .help-button{
            position:fixed; right:25px; bottom:25px; z-index:9999;
            background:#8B4513; color:#FFD700; padding:16px 22px; font-size:17px;
            border-radius:18px; font-weight:900; box-shadow:0 8px 22px rgba(0,0,0,0.2); cursor:pointer;
            border:0;
        }
        .help-panel{
            position:fixed; right:25px; bottom:100px; z-index:9998;
            width:520px; max-height:70vh; overflow-y:auto;
            background: linear-gradient(135deg, #FFFDF5 0%, #F8EAD9 100%);
            border:1px solid #CFB53B; border-radius:12px; padding:20px; box-shadow:0 14px 34px rgba(0,0,0,0.22);
            color:#4a2f1f; transform: translateY(12px); opacity:0; visibility:hidden; transition: all .22s ease-in-out;
        }
        #help-toggle:checked + label.help-button + .help-panel{
            opacity:1; visibility:visible; transform: translateY(0);
        }
        .help-panel .head{display:flex; justify-content:space-between; align-items:center; margin-bottom:12px}
        .help-panel .title{font-weight:900; color:#654321; font-size:16px}
        .help-panel .help-close{background:#8B4513; color:#FFD700; padding:6px 10px; border-radius:8px; cursor:pointer; font-weight:800}
        .help-body{font-size:14.5px; color:#4a2f1f; line-height:1.5}
        .help-body b {color: #654321;}
        .help-body ul { padding-left: 20px; }
        .help-body li { margin-bottom: 8px; }

        /* compact recent blends styles - improved font sizes */
        .recent-compact { padding-left:6px; padding-right:6px; }
        .compact-card{
            background: linear-gradient(180deg,#FFF8E1 0%, #FFF6EA 100%);
            border:1px solid #E3C77A; border-radius:8px; padding:10px; margin-bottom:8px; color:#654321;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .compact-top{display:flex; justify-content:space-between; align-items:center; margin-bottom:8px}
        .compact-name{font-weight:800; font-size:15px}
        .compact-ts{font-size:12px; color:#8B4513; opacity:0.95; font-weight:700}
        .comp-pills{font-size:12.5px; margin-bottom:8px}
        .comp-pill{
            display:inline-block; padding:3px 8px; margin-right:6px; margin-bottom: 4px; border-radius:999px;
            background:rgba(139,69,19,0.06); border:1px solid rgba(139,69,19,0.12);
            font-weight:700; color:#654321;
        }
        .props-inline{
            font-size:12px; color:#4a2f1f; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
        }
        .props-inline small{ font-size:11px; color:#4a2f1f; opacity:0.95; margin-right:8px; }
        </style>

        <input id="help-toggle" type="checkbox" />
        <label for="help-toggle" class="help-button">💬 App Guide</label>

        <div class="help-panel" aria-hidden="true">
            <div class="head">
                <div class="title">Welcome to the Eagle Blend Optimizer!</div>
                <label for="help-toggle" class="help-close">Close</label>
            </div>
            <div class="help-body">
                <p>This is your central hub for AI-powered fuel blend analysis, prediction, and optimization. The app is organized into several powerful tabs:</p>
                <ul>
                    <li><b>📊 Dashboard:</b> You are here! This is your main overview, showing key metrics like model accuracy, recent app activity, and the highest cost savings achieved. The list on the right gives you a live look at the most recently created blends.</li>
                    <li><b>🎛️ Blend Designer:</b> This is your creative sandbox. Manually define the fractions and properties of up to five components to instantly predict the final properties of a new blend. You can also switch to <b>Batch Mode</b> to upload a CSV and predict many blends at once.</li>
                    <li><b>⚙️ Optimization Engine:</b> Go beyond simple prediction. Here, you set the <b>target properties</b> you want to achieve. The AI engine will then run an optimization to find the ideal component fractions that best meet your goals and constraints, such as minimizing cost.</li>
                    <li><b>📤 Blend Comparison:</b> This is your analysis workbench. Select up to three previously saved blends from your database to perform a detailed side-by-side comparison. The charts will help you visualize differences in their cost, composition, and performance profiles.</li>
                    <li><b>📚 Fuel Registry:</b> The heart of your data. This tab is where you manage the database of all raw <b>Components</b> and saved <b>Blends</b>. You can view, add, and delete records here.</li>
                    <li><b>🧠 Model Insights:</b> Look under the hood of the AI. This tab shows detailed performance metrics for the prediction model, helping you understand its accuracy and where its predictions are most reliable.</li>
                </ul>
                <hr style="border-top: 1px solid #CFB53B; margin: 15px 0;">
                <p><b>Getting Started:</b> A great first step is to visit the <b>Fuel Registry</b> to see your available components, then head to the <b>Blend Designer</b> to create your first prediction!</p>
            </div>
        </div>
    """, unsafe_allow_html=True)



# ---------- Floating "How to Use" (bigger button + inline content) + compact CSS ----------


    # ---------- Main split (adjusted for better balance) ----------
    left_col, right_col = st.columns([0.55, 0.45])

    # --- LEFT: Blend entries line chart ---
    with left_col:
        # FIXED: Replaced st.subheader with styled markdown for consistent color
        st.markdown('<h2 style="color:#4a2f1f; font-size:1.75rem;">Blend Entries Per Day</h2>', unsafe_allow_html=True)

        # Using DUMMY DATA as per original snippet for illustration
        today = pd.Timestamp.today().normalize()
        dates = pd.date_range(end=today, periods=14)
        ddf = pd.DataFrame({"day": dates, "Blends": np.array([2,3,1,5,6,2,4,9,3,4,2,1,5,6])})

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=ddf["day"], y=ddf["Blends"],
            mode="lines+markers", line=dict(width=3, color="#8B4513"),
            marker=dict(size=6), name="Blends"
        ))
        fig_daily.add_trace(go.Scatter(
            x=ddf["day"], y=ddf["Blends"],
            mode="lines", line=dict(width=0), fill="tozeroy",
            fillcolor="rgba(207,181,59,0.23)", showlegend=False
        ))
        fig_daily.update_layout(
            title="Recent Blend Creation (preview)",
            xaxis_title="Date", yaxis_title="Number of Blends",
            plot_bgcolor="white", paper_bgcolor="white", # Set background to white
            margin=dict(t=40, r=10, b=36, l=50), # Tighter margins
            font=dict(color="#4a2f1f") # Ensure text color is not white
        )
        fig_daily.update_xaxes(gridcolor="rgba(139,69,19,0.12)", tickfont=dict(color="#654321"))
        fig_daily.update_yaxes(gridcolor="rgba(139,69,19,0.12)", tickfont=dict(color="#654321"))
        st.plotly_chart(fig_daily, use_container_width=True)

        # st.caption("Chart preview uses dummy data. To show live counts, uncomment the LIVE DATA block in the code.")

    # --- RIGHT: Compact Recent Blends (with larger fonts and clear timestamp) ---
    with right_col:
        st.markdown('<div class="recent-compact">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1.15rem; font-weight:800; color:#654321; margin-bottom:12px;">🗒️ Recent Blends</div>', unsafe_allow_html=True)

        df_recent = overview['last_blends']   #get("last_blends", pd.DataFrame())
        if df_recent is None or df_recent.empty:
            st.info("No blends yet. Start blending today!")
        else:
            if "created_at" in df_recent.columns and not pd.api.types.is_datetime64_any_dtype(df_recent["created_at"]):
                with pd.option_context('mode.chained_assignment', None):
                    df_recent["created_at"] = pd.to_datetime(df_recent["created_at"], errors="coerce")

            for _, row in df_recent.iterrows():
                name = str(row.get("blend_name", "Untitled"))
                created = row.get("created_at", "")
                ts = "" if pd.isna(created) else pd.to_datetime(created).strftime("%Y-%m-%d %H:%M:%S")

                comp_html = ""
                for i in range(1, 6):
                    key = f"Component{i}_fraction"
                    val = row.get(key)
                    if val is None or (isinstance(val, float) and math.isnan(val)) or val == 0:
                        continue
                    comp_html += f'<span class="comp-pill">C{i}: {float(val)*100:.0f}%</span>'

                props = []
                for j in range(1, 11):
                    pj = row.get(f"BlendProperty{j}")
                    if pj is not None and not (isinstance(pj, float) and math.isnan(pj)):
                        props.append(f"P{j}:{float(pj):.3f}")
                props_html = " · ".join(props) if props else "No properties available."


                st.markdown(f"""
                    <div class="compact-card">
                        <div class="compact-top">
                            <div class="compact-name">{name}</div>
                            <div class="compact-ts">{ts}</div>
                        </div>
                        <div class="comp-pills">{comp_html}</div>
                        <div class="props-inline"><small>{props_html}</small></div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------------------------------------------------
#                                                            Blend Designer Tab                                                  
# ----------------------------------------------------------------------------------------------------------------------------------------------

# --- Add these new functions to your functions section ---

@st.cache_data
def get_components_from_db(db_path="eagleblend.db") -> pd.DataFrame:
    """Fetches component data, sorted by the most recent entries."""
    with sqlite3.connect(db_path) as conn:
        # Assuming 'id' or a timestamp column indicates recency. Let's use 'id'.
        query = "SELECT * FROM components ORDER BY id DESC"
        df = pd.read_sql_query(query, conn)
    return df

def log_activity(activity_type: str, details: str = "", db_path="eagleblend.db"):
    """Logs an activity to the activity_log table."""
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cur.execute(
                "INSERT INTO activity_log (timestamp, activity_type) VALUES (?, ?)",
                (timestamp, activity_type)
            )
            conn.commit()
    except Exception as e:
        st.error(f"Failed to log activity: {e}")

# Instantiate the predictor once
@st.cache_resource
def load_model():
    from predictor import EagleBlendPredictor
    # heavy model load...
    return EagleBlendPredictor()

if 'predictor' not in st.session_state:
    st.session_state.predictor = load_model()

with tabs[1]:
    # --- State Initialization ---
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'preopt_cost' not in st.session_state:
        st.session_state.preopt_cost = 0.0
    if 'last_input_data' not in st.session_state:
        st.session_state.last_input_data = {}

    # --- Prediction & Saving Logic ---
    def handle_prediction():
        """
        Gathers data from UI, formats it, runs prediction, and stores results.
        """
        log_activity("prediction", "User ran a new blend prediction.")
        
        fractions = []
        properties_by_comp = [[] for _ in range(5)]
        unit_costs = []

        # 1. Gather all inputs from session state
        for i in range(5):
            frac = st.session_state.get(f"c{i}_fraction", 0.0)
            fractions.append(frac)
            unit_costs.append(st.session_state.get(f"c{i}_cost", 0.0))
            for j in range(1, 11):
                prop = st.session_state.get(f"c{i}_prop{j}", 0.0)
                properties_by_comp[i].append(prop)

        # 2. Validate weights
        if abs(sum(fractions) - 1.0) > 0.01:
            st.warning("⚠️ Total of component fractions must sum to 1.0.")
            st.session_state.prediction_made = False
            return
            
        # 3. Format DataFrame for the model
        model_input_data = {"blend_name": [st.session_state.get("blend_name", "Untitled Blend")]}
        # Add fractions first
        for i in range(5):
            model_input_data[f'Component{i+1}_fraction'] = [fractions[i]]
        # Add properties in the required order (interleaved)
        for j in range(10): # Property1, Property2, ...
            for i in range(5): # Component1, Component2, ...
                col_name = f'Component{i+1}_Property{j+1}'
                model_input_data[col_name] = [properties_by_comp[i][j]]

        df_model = pd.DataFrame(model_input_data)
        
        # 4. Run prediction
        predictor = st.session_state.predictor
        # results = predictor.predict_all(df_model.drop(columns=['blend_name']))
        # st.session_state.prediction_results = results[0]  # Get the first (and only) row of results
        # --- FIX: Handles DataFrame output and converts it to an array for single prediction ---
        results_df = predictor.predict_all(df_model.drop(columns=['blend_name']))
        st.session_state.prediction_results = results_df.iloc[0].values
        
        # --- Conditional cost calculation ---
        # 5. Calculate cost only if all unit costs are provided and greater than zero
        if all(c > 0.0 for c in unit_costs):
            st.session_state.preopt_cost = sum(f * c for f, c in zip(fractions, unit_costs))
            st.session_state.cost_calculated = True
        else:
            st.session_state.preopt_cost = 0.0
            st.session_state.cost_calculated = False
        # st.session_state.preopt_cost = sum(f * c for f, c in zip(fractions, unit_costs))
        
        # 6. Store inputs for saving/downloading
        st.session_state.last_input_data = model_input_data
        
        st.session_state.prediction_made = True
        st.success("Prediction complete!")

# def handle_prediction():
#     """
#     Gathers data from UI, formats it, runs prediction, and stores results.
#     """
#     start_time = time.time() # Start the timer
#     log_activity("prediction", "User ran a new blend prediction.")
    
#     fractions = []
#     properties_by_comp = [[] for _ in range(5)]
#     unit_costs = []

#     # 1. Gather all inputs from session state
#     for i in range(5):
#         frac = st.session_state.get(f"c{i}_fraction", 0.0)
#         fractions.append(frac)
#         unit_costs.append(st.session_state.get(f"c{i}_cost", 0.0))
#         for j in range(1, 11):
#             prop = st.session_state.get(f"c{i}_prop{j}", 0.0)
#             properties_by_comp[i].append(prop)

#     # 2. Validate weights
#     if abs(sum(fractions) - 1.0) > 0.01:
#         st.warning("⚠️ Total of component fractions must sum to 1.0.")
#         st.session_state.prediction_made = False
#         return
        
#     # 3. Format DataFrame for the model
#     model_input_data = {"blend_name": [st.session_state.get("blend_name", "Untitled Blend")]}
#     for i in range(5):
#         model_input_data[f'Component{i+1}_fraction'] = [fractions[i]]
#     for j in range(10):
#         for i in range(5):
#             col_name = f'Component{i+1}_Property{j+1}'
#             model_input_data[col_name] = [properties_by_comp[i][j]]

#     df_model = pd.DataFrame(model_input_data)
    
#     # 4. Run prediction
#     predictor = st.session_state.predictor
#     results_df = predictor.predict_all(df_model.drop(columns=['blend_name']))
#     st.session_state.prediction_results = results_df.iloc[0].values
    
#     # 5. Calculate cost
#     if all(c > 0.0 for c in unit_costs):
#         st.session_state.preopt_cost = sum(f * c for f, c in zip(fractions, unit_costs))
#         st.session_state.cost_calculated = True
#     else:
#         st.session_state.preopt_cost = 0.0
#         st.session_state.cost_calculated = False
    
#     # 6. Store inputs for saving/downloading
#     st.session_state.last_input_data = model_input_data
    
#     st.session_state.prediction_made = True
    
#     # --- FIX: Stop the timer and create the new success message ---
#     end_time = time.time()
#     duration = end_time - start_time
#     st.success(f"✅ Prediction complete in {duration:.2f} seconds! Scroll down to see the results.")

def handle_save_prediction():
    """Formats the last prediction's data and saves it to the database."""
    if not st.session_state.get('prediction_made', False):
        st.error("Please run a prediction before saving.")
        return

    # Prepare DataFrame in the format expected by `add_blends`
    # save_df_data = st.session_state.last_input_data.copy()
    # --- FIX: This gets the most recent blend name before saving ---
    save_df_data = st.session_state.last_input_data.copy()
    save_df_data['blend_name'] = [st.session_state.get('blend_name', 'Untitled Blend')]
    
    # Add blend properties and cost
    for i, prop_val in enumerate(st.session_state.prediction_results, 1):
        save_df_data[f'BlendProperty{i}'] = [prop_val]
    
    save_df_data['PreOpt_Cost'] = [st.session_state.preopt_cost]
    
    # Add unit costs
    for i in range(5):
        save_df_data[f'Component{i+1}_unit_cost'] = st.session_state.get(f'c{i}_cost', 0.0)
        
    save_df = pd.DataFrame(save_df_data)

    try:
        result = add_blends(save_df)
        log_activity("save_prediction", f"Saved blend: {save_df['blend_name'].iloc[0]}")
        get_all_blends_data.clear()
        st.success(f"Successfully saved blend '{save_df['blend_name'].iloc[0]}' to the database!")
    except Exception as e:
        st.error(f"Failed to save blend: {e}")


    # --- UI Rendering ---
    col_header = st.columns([0.8, 0.2])
    with col_header[0]:
        st.subheader("🎛️ Blend Designer")
    with col_header[1]:
        batch_blend = st.checkbox("Batch Blend Mode", value=False, key="batch_blend_mode")

    # --- This is the new, fully functional batch mode block ---
    if batch_blend:
        st.subheader("📤 Batch Processing")
        st.markdown("Upload a CSV file with blend recipes to predict their properties in bulk. The file must contain the 55 feature columns required by the model.")

        # Provide a template for download
        # NOTE: You will need to create a dummy CSV file named 'batch_template.csv'
        # with the 55 required column headers for this to work.
        try:
            with open("assets/batch_template.csv", "rb") as f:
                st.download_button(
                    label="📥 Download Batch Template (CSV)",
                    data=f,
                    file_name="batch_template.csv",
                    mime="text/csv"
                )
        except FileNotFoundError:
            st.warning("Batch template file not found. Please create 'assets/batch_template.csv'.")


        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="batch_upload")

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.markdown("##### Uploaded Data Preview")
                st.dataframe(input_df.head())

                if st.button("🧪 Run Batch Prediction", use_container_width=True, type="primary"):
                    # Basic validation: check for at least the fraction columns
                    required_cols = [f'Component{i+1}_fraction' for i in range(5)]
                    if not all(col in input_df.columns for col in required_cols):
                        st.error(f"Invalid file format. The uploaded CSV is missing one or more required columns like: {', '.join(required_cols)}")
                    else:
                        with st.spinner("Running batch prediction... This may take a moment."):
                            # Run prediction on the entire DataFrame
                            predictor = st.session_state.predictor
                            results_df = predictor.predict_all(input_df)
                            
                            # Combine original data with the results
                            # Ensure column names for results are clear
                            results_df.columns = [f"BlendProperty{i+1}" for i in range(results_df.shape[1])]
                            
                            # Combine input and output dataframes
                            final_df = pd.concat([input_df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
                            
                            st.session_state['batch_results'] = final_df
                            st.success("Batch prediction complete!")

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

        # Display results and download button if they exist in the session state
        if 'batch_results' in st.session_state:
            st.markdown("---")
            st.subheader("✅ Batch Prediction Results")
            
            results_to_show = st.session_state['batch_results']
            st.dataframe(results_to_show)

            csv_data = results_to_show.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv_data,
                file_name="batch_prediction_results.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        # --- Manual Blend Designer UI ---
        all_components_df = get_components_from_db()
        # st.text_input("Blend Name", "My New Blend", key="blend_name", help="Give your blend a unique name before saving.")
        # st.markdown("---")

        for i in range(5):
            # Unique keys for each widget within the component expander
            select_key = f"c{i}_select"
            name_key = f"c{i}_name"
            frac_key = f"c{i}_fraction"
            cost_key = f"c{i}_cost"

            # Check if a selection from dropdown was made
            if select_key in st.session_state and st.session_state[select_key] != "---":
                selected_name = st.session_state[select_key]
                comp_data = all_components_df[all_components_df['component_name'] == selected_name].iloc[0]
                
                # Auto-populate session state values
                st.session_state[name_key] = comp_data['component_name']
                st.session_state[frac_key] = comp_data.get('component_fraction', 0.2)
                # st.session_state[cost_key] = comp_data.get('unit_cost', 0.0)
                # --- Handle missing unit_cost from DB correctly ---
                cost_val = comp_data.get('unit_cost', 0.0)
                st.session_state[cost_key] = 0.0 if pd.isna(cost_val) else float(cost_val)
                for j in range(1, 11):
                    prop_key = f"c{i}_prop{j}"
                    st.session_state[prop_key] = comp_data.get(f'property{j}', 0.0)
                
                # Reset selectbox to avoid re-triggering
                st.session_state[select_key] = "---"

            with st.expander(f"**Component {i+1}**", expanded=(i==0)):
                # --- This is the placeholder for your custom filter ---
                # Example: Only show components ending with a specific number
                # filter_condition = all_components_df['component_name'].str.endswith(str(i + 1))
                # For now, we show all components
                filter_condition = pd.Series([True] * len(all_components_df), index=all_components_df.index)
                
                filtered_df = all_components_df[filter_condition]
                #component_options = ["---"] + filtered_df['component_name'].tolist() 
                # component_options = ["---"] + [m for m in filtered_df['component_name'].tolist() if  m.endswith(f"Component_{i+1}") ]  
                options = filter_component_options(all_components_df, i)
                component_options = ["---"] + options

                st.selectbox(
                    "Load from Registry",
                    options=component_options,
                    key=select_key,
                    help="Select a saved component to auto-populate its properties."
                )

                c1, c2, c3 = st.columns([1.5, 2, 2])
                with c1:
                    st.text_input("Component Name", key=name_key)
                    st.number_input("Fraction", min_value=0.0, max_value=1.0, step=0.01, key=frac_key, format="%.3f")
                    st.number_input("Unit Cost ($)", min_value=0.0, step=0.01, key=cost_key, format="%.2f")
                with c2:
                    for j in range(1, 6):
                        st.number_input(f"Property {j}", key=f"c{i}_prop{j}", format="%.4f")
                with c3:
                    for j in range(6, 11):
                        st.number_input(f"Property {j}", key=f"c{i}_prop{j}", format="%.4f")
        
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        # st.button("🧪 Predict Blended Properties", on_click=handle_prediction, use_container_width=True, type="primary")
        # --- FIX: Changed button call to prevent page jumping ---
        # --- In the "Manual Blend Designer UI" section ---
        if st.button("🧪 Predict Blended Properties", use_container_width=False, type="primary"):
            with st.spinner("🧠 Running prediction... Please wait."):
                handle_prediction()

        # --- Results Section ---
        if st.session_state.get('prediction_made', False):
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            st.subheader("📈 Prediction Results")

            results_array = st.session_state.get('prediction_results', np.zeros(10))

            # Display the 10 Property KPI cards
            kpi_cols = st.columns(5)
            for i in range(10):
                with kpi_cols[i % 5]:
                    st.markdown(f"""
                        <div class="metric-card" style="margin-bottom: 10px; padding: 0.8rem;">
                            <div class="metric-label" style="font-size: 0.8rem;">Blend Property {i+1}</div>
                            <div class="metric-value" style="font-size: 1.5rem;">{results_array[i]:.4f}</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Display the Centered, smaller cost KPI card
            _, mid_col, _ = st.columns([1.5, 2, 1.5])
            with mid_col:
                cost_val = st.session_state.get('preopt_cost', 0.0)
                cost_calculated = st.session_state.get('cost_calculated', False)
                if cost_calculated:
                    cost_display = f"${cost_val:,.2f}"
                    delta_text = "Per unit fuel"
                else:
                    cost_display = "N/A"
                    delta_text = "Enter all component costs to calculate"
                
                st.markdown(f"""
                    <div class="metric-card" style="border-color: #8B4513; background: #FFF8E1; padding: 0.8rem;">
                        <div class="metric-label" style="font-size: 0.8rem;">Predicted Blend Cost</div>
                        <div class="metric-value" style="font-size: 1.5rem;">{cost_display}</div>
                        <div class="metric-delta" style="font-size: 0.8rem;">{delta_text}</div>
                    </div>
                """, unsafe_allow_html=True)

            # --- Visualizations & Actions Section ---
            st.subheader("📊 Visualizations & Actions")
            vis_col1, vis_col2 = st.columns(2)

            with vis_col1:
                # Pie Chart
                fractions = [st.session_state.get(f"c{i}_fraction", 0.0) for i in range(5)]
                labels = [st.session_state.get(f"c{i}_name", f"Component {i+1}") for i in range(5)]
                pie_fig = px.pie(
                    values=fractions, names=labels, title="Component Fractions",
                    hole=0.4, color_discrete_sequence=px.colors.sequential.YlOrBr_r
                )
                pie_fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(pie_fig, use_container_width=True)

                # --- This is the ONE AND ONLY 'blend_name' input ---
                st.text_input(
                    "Blend Name for Saving", 
                    "My New Blend", 
                    key="blend_name", 
                    help="Give your blend a unique name before saving."
                )

            with vis_col2:
                # Bar Chart
                prop_to_view = st.selectbox(
                    "Select Property to Visualize",
                    options=[f"Property{j}" for j in range(1, 11)],
                    key="viz_property_select"
                )
                prop_idx = int(prop_to_view.replace("Property", "")) - 1
                bar_values = [st.session_state.get(f"c{i}_prop{prop_idx+1}", 0.0) for i in range(5)]
                blend_prop_value = results_array[prop_idx]
                bar_labels = [f"Comp {i+1}" for i in range(5)] + ["Blend"]
                all_values = bar_values + [blend_prop_value]
                bar_df = pd.DataFrame({"Component": bar_labels, "Value": all_values})
                
                # --- Lighter brown color for the bars ---
                bar_colors = ['#A67C52'] * 5 + ['#654321'] 

                bar_fig = px.bar(bar_df, x="Component", y="Value", title=f"Comparison for {prop_to_view}")
                bar_fig.update_traces(marker_color=bar_colors)
                bar_fig.update_layout(showlegend=False)
                st.plotly_chart(bar_fig, use_container_width=True)

                # Download button is aligned here
                download_df = pd.DataFrame(st.session_state.last_input_data)
                file_name = st.session_state.get('blend_name', 'blend_results').replace(' ', '_')
                for i in range(5):
                    download_df[f'Component{i+1}_unit_cost'] = st.session_state.get(f'c{i}_cost', 0.0)
                for i, res in enumerate(results_array, 1):
                    download_df[f'BlendProperty{i}'] = res
                csv_data = download_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"{file_name}.csv",
                    mime='text/csv',
                    use_container_width=True,
                    help="Download all inputs and predicted outputs to a CSV file."
                )
            
            # --- This is the ONE AND ONLY 'Save' button ---
            if st.button("💾 Save Prediction to Database", use_container_width=False):
                handle_save_prediction()
            # This empty markdown is a trick to add vertical space
    st.markdown('<div style="height: 36px;"></div>', unsafe_allow_html=True)

    # --- Floating "How to Use" button ---
    st.markdown("""
        <style>
            /* Styles for the help panel and button */
            #help-toggle-designer{display:none;}
            .help-button-designer{
                position:fixed; right:25px; bottom:25px; z-index:9999;
                background:#8B4513; color:#FFD700; padding:16px 22px; font-size:17px;
                border-radius:18px; font-weight:900; box-shadow:0 8px 22px rgba(0,0,0,0.2);
                cursor:pointer; border:0;
            }
            .help-panel-designer{
                position:fixed; right:25px; bottom:100px; z-index:9998;
                width:520px; max-height:70vh; overflow-y:auto;
                background: linear-gradient(135deg, #FFFDF5 0%, #F8EAD9 100%);
                border:1px solid #CFB53B; border-radius:12px; padding:20px; box-shadow:0 14px 34px rgba(0,0,0,0.22);
                color:#4a2f1f; transform: translateY(12px); opacity:0; visibility:hidden; transition: all .22s ease-in-out;
            }
            #help-toggle-designer:checked + label.help-button-designer + .help-panel-designer{
                opacity:1; visibility:visible; transform: translateY(0);
            }
            /* Style for the header and close button inside the panel */
            .help-panel-designer .head{display:flex; justify-content:space-between; align-items:center; margin-bottom:12px}
            .help-panel-designer .title{font-weight:900; color:#654321; font-size:16px}
            .help-panel-designer .help-close{background:#8B4513; color:#FFD700; padding:6px 10px; border-radius:8px; cursor:pointer; font-weight:800}
        </style>
        <input id="help-toggle-designer" type="checkbox" />
        <label for="help-toggle-designer" class="help-button-designer">💬 Help</label>
        <div class="help-panel-designer">
            <div class="head">
                <div class="title">Using the Blend Designer</div>
                <label for="help-toggle-designer" class="help-close">Close</label>
            </div>
        <p style="margin-top:0;">This tab is your creative sandbox for designing and predicting fuel properties. It has two modes:</p>
        
        <b style="color: #654321;">Manual Mode (Default):</b>
            <ul style="padding-left: 20px; list-style-position: outside; margin-bottom:0;">
                <li style="margin-bottom: 8px;"><b>Configure:</b> Define up to five components. Use the 'Load from Registry' dropdown to auto-fill data or enter properties manually.</li>
                <li style="margin-bottom: 8px;"><b>Predict:</b> Once component fractions sum to 1.0, click <b>Predict</b>. The AI calculates the blend's 10 properties and its cost.</li>
                <li style="margin-bottom: 8px;"><b>Analyze:</b> Two charts appear after prediction. The <b>Pie Chart</b> shows the component mix. The <b>Bar Chart</b> compares each component's property to the final blend's.</li>
                <li style="margin-bottom: 8px;"><b>Save:</b> After predicting, enter a unique name and save the blend to the database.</li>
            </ul>
                 <p style="margin-top:15px;"><b style="color: #654321;">Batch Blend Mode:</b></p>
                <ul style="padding-left: 20px; list-style-position: outside; margin-top:0;">
                    <li style="margin-bottom: 8px;"><b>Activate:</b> Toggle on Batch Mode to predict many recipes at once.</li>
                    <li style="margin-bottom: 8px;"><b>Process:</b> Download the CSV template, fill it with your data, upload it, and click 'Run Batch Prediction'.</li>
                    <li style="margin-bottom: 8px;"><b>Download:</b> The results for all your blends will appear in a table, ready to download.</li>
                </ul>
        </div>
    """, unsafe_allow_html=True)


## ----------------------------------------------------------------------------------------------------------------------------------------------
##                                                   Optimization Engine Tab
##-----------------------------------------------------------------------------------------------------------------------------------------------

with tabs[2]:
    st.subheader("⚙️ Optimization Engine")
    st.markdown("Define your property goals, select base components, and run the optimizer to find the ideal blend recipe.")

    # --- State Initialization ---
    if 'optimization_running' not in st.session_state:
        st.session_state.optimization_running = False
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'optimization_time' not in st.session_state:
        st.session_state.optimization_time = 0.0

    # --- Optimization Goals ---
    st.markdown("#### 1. Define Optimization Goals")
    
    # Using a container to group the goal inputs
    with st.container(border=True):
        cols_row1 = st.columns(5)
        cols_row2 = st.columns(5)
        
        for i in range(1, 11):
            col = cols_row1[(i-1)] if i <= 5 else cols_row2[(i-6)]
            with col:
                st.number_input(f"Property {i}", key=f"opt_target_{i}", value=0.0, step=0.01, format="%.4f")
                st.toggle("Fix Target", key=f"opt_fix_{i}", help=f"Toggle on to make Property {i} a fixed constraint.")

    # --- Component Selection (Copied and Adapted) ---
    st.markdown("#### 2. Select Initial Components")
    all_components_df_opt = get_components_from_db() # Use a different variable to avoid conflicts
    
    main_cols = st.columns(2)
    with main_cols[0]: # Left side for first 3 components
        for i in range(3):
            with st.expander(f"**Component {i+1}**", expanded=(i==0)):
                # Auto-population and input fields logic (reused from Blend Designer)
                # Note: Keys are prefixed with 'opt_' to ensure they are unique to this tab
                select_key, name_key, frac_key, cost_key = f"opt_c{i}_select", f"opt_c{i}_name", f"opt_c{i}_fraction", f"opt_c{i}_cost"
                
                # Auto-population logic...
                if select_key in st.session_state and st.session_state[select_key] != "---":
                    selected_name = st.session_state[select_key]
                    comp_data = all_components_df_opt[all_components_df_opt['component_name'] == selected_name].iloc[0]
                    st.session_state[name_key] = comp_data['component_name']
                    st.session_state[frac_key] = comp_data.get('component_fraction', 0.2)
                    cost_val = comp_data.get('unit_cost', 0.0)
                    st.session_state[cost_key] = 0.0 if pd.isna(cost_val) else float(cost_val)
                    for j in range(1, 11):
                        st.session_state[f"opt_c{i}_prop{j}"] = comp_data.get(f'property{j}', 0.0)
                    st.session_state[select_key] = "---"

                # UI for component
                # component_options = ["---"] + all_components_df_opt['component_name'].tolist()
                options = filter_component_options(all_components_df_opt, i)
                component_options = ["---"] + options
                st.selectbox("Load from Registry", options=component_options, key=select_key)
                c1, c2, c3 = st.columns([1.5, 2, 2])
                with c1:
                    st.text_input("Component Name", key=name_key)
                    st.number_input("Unit Cost ($)", min_value=0.0, step=0.01, key=cost_key, format="%.2f")
                with c2:
                    for j in range(1, 6): st.number_input(f"Property {j}", key=f"opt_c{i}_prop{j}", format="%.4f")
                with c3:
                    for j in range(6, 11): st.number_input(f"Property {j}", key=f"opt_c{i}_prop{j}", format="%.4f")

    with main_cols[1]: # Right side for last 2 components and controls
        for i in range(3, 5):
             with st.expander(f"**Component {i+1}**", expanded=False):
                # Auto-population and input fields logic...
                select_key, name_key, frac_key, cost_key = f"opt_c{i}_select", f"opt_c{i}_name", f"opt_c{i}_fraction", f"opt_c{i}_cost"
                if select_key in st.session_state and st.session_state[select_key] != "---":
                    selected_name = st.session_state[select_key]
                    comp_data = all_components_df_opt[all_components_df_opt['component_name'] == selected_name].iloc[0]
                    st.session_state[name_key] = comp_data['component_name']
                    st.session_state[frac_key] = comp_data.get('component_fraction', 0.2)
                    cost_val = comp_data.get('unit_cost', 0.0)
                    st.session_state[cost_key] = 0.0 if pd.isna(cost_val) else float(cost_val)
                    for j in range(1, 11):
                        st.session_state[f"opt_c{i}_prop{j}"] = comp_data.get(f'property{j}', 0.0)
                    st.session_state[select_key] = "---"
                # component_options = ["---"] + all_components_df_opt['component_name'].tolist()
                options = filter_component_options(all_components_df_opt, i)
                component_options = ["---"] + options
                st.selectbox("Load from Registry", options=component_options, key=select_key)
                c1, c2, c3 = st.columns([1.5, 2, 2])
                with c1:
                    st.text_input("Component Name", key=name_key)
                    st.number_input("Unit Cost ($)", min_value=0.0, step=0.01, key=cost_key, format="%.2f")
                with c2:
                    for j in range(1, 6): st.number_input(f"Property {j}", key=f"opt_c{i}_prop{j}", format="%.4f")
                with c3:
                    for j in range(6, 11): st.number_input(f"Property {j}", key=f"opt_c{i}_prop{j}", format="%.4f")

        # --- Optimization Controls ---
        with st.container(border=True):
            st.markdown("##### 3. Configure & Run")
            st.checkbox("Include Cost in Optimization", value=True, key="opt_include_cost")
            # ... inside the "Configure & Run" container ...

            st.slider(
                "Optimization Steps (Generations)",
                min_value=10, max_value=100, value=20, key="opt_generations",
                help="Controls how many iterations the algorithm runs. Higher is slower but finds better solutions."
            )
            st.slider(
                "Optimization Depth (Population Size)",
                min_value=10, max_value=500, value=10, key="opt_pop_size",
                help="Controls how many candidate solutions are tested in each step. Higher is slower but explores more options."
            )

            run_button_col, spinner_col = st.columns([3, 1])
            # ... rest of the container code ...
            
            # Run button and spinner logic
            run_button_col, spinner_col = st.columns([3, 1])
            with run_button_col:
                # New Code:
                if st.button("🚀 Run Optimization", use_container_width=False, type="primary", disabled=st.session_state.optimization_running):
                    st.session_state.optimization_running = True
                    log_activity("optimization")

                    start_time = time.time()
                    
                    # --- FIX: Create a placeholder for the progress bar ---
                    progress_placeholder = st.empty()
                    
                    # Gather data for the optimization function
                    targets = {f"Property{i}": st.session_state.get(f"opt_target_{i}", 0.0) for i in range(1, 11)}
                    fixed_targets = {f"Property{i}": targets[f"Property{i}"] for i in range(1, 11) if st.session_state.get(f"opt_fix_{i}", False)}
                    
                    include_cost = st.session_state.get('opt_include_cost', True)
                    generations = st.session_state.get('opt_generations', 20)
                    pop_size = st.session_state.get('opt_pop_size', 20)
                    
                    # Initialize the progress bar in the placeholder
                    progress_bar = progress_placeholder.progress(0, text="Initializing Optimization...")
                    
                    # Call the function, passing the progress_bar object
                    st.session_state.optimization_results = run_real_optimization(
                        targets, fixed_targets, None, include_cost, generations, pop_size, progress_bar
                    )
                    
                    # Update the bar to 100% and show a completion message
                    progress_bar.progress(1.0, text="Optimization Complete!")
                    time.sleep(1.5) # Optional: pause for a moment to show completion
                    progress_placeholder.empty() # Clear the progress bar from the screen

                    st.session_state.optimization_time = time.time() - start_time
                    st.session_state.optimization_running = False
                    st.rerun()

            with spinner_col:
                if st.session_state.optimization_running:
                    st.markdown('<div class="spinner"></div>', unsafe_allow_html=True)

            if st.session_state.optimization_time > 0:
                st.success(f"Optimization complete in {st.session_state.optimization_time:.2f} seconds. Scroll down to see Results")

    # --- Results Section ---
    if st.session_state.optimization_results:
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.subheader("🏆 Optimization Results")

        results = st.session_state.optimization_results
        
        # --- FIX: Add sorting controls ---
        st.markdown("##### Sort Solutions By")
        sort_option = st.radio(
            "Sort Solutions By",
            options=["Best Quality (Lowest Error)", "Lowest Cost", "Best Quality Score"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # Dynamically sort the results list based on the selected option
        if sort_option == "Lowest Cost":
            sorted_results = sorted(results, key=lambda x: x.get('optimized_cost', float('inf')))
        elif sort_option == "Best Quality Score":
            # Calculate score for each result before sorting
            for res in results:
                res['quality_score'] = calculate_quality_score(res.get("error"))
            sorted_results = sorted(results, key=lambda x: x.get('quality_score', 0), reverse=True)
        else: # Default sort by error
            sorted_results = results # Already pre-sorted by the function

        # --- FIX: Populate dropdown with all sorted results ---
        # The first item in the list is now always the "best" according to the sort
        result_options = {
            i: f"Solution {i+1} (Error: {res['error']:.4f}, Cost: ${res.get('optimized_cost', 0):.2f})" 
            for i, res in enumerate(sorted_results)
        }
        
        st.markdown("##### Select Solution to View")
        selected_idx = st.selectbox(
            "Select Solution to View", 
            options=list(result_options.keys()), 
            format_func=lambda x: result_options[x],
            label_visibility="collapsed"
        )
        
        # The rest of the UI will automatically update based on the selected solution
        selected_solution = sorted_results[selected_idx]


        # --- New Layout for Component Fractions (Centered) ---
        st.markdown("##### Optimal Component Fractions")
        _, c1, c2, c3, c4, c5, _ = st.columns([0.5, 1, 1, 1, 1, 1, 0.5])
        cols = [c1, c2, c3, c4, c5]
        for i, frac in enumerate(selected_solution["component_fractions"]):
            with cols[i]:
                comp_name = st.session_state.get(f"opt_c{i}_name") or f"Component {i+1}"
                st.markdown(f"""
                <div class="metric-card" style="padding: 0.8rem;">
                    <div class="metric-label" style="font-size: 0.8rem;">{comp_name}</div>
                    <div class="metric-value" style="font-size: 1.5rem;">{frac*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div style="height:15px;"></div>', unsafe_allow_html=True) # Spacer

        # --- New Layout for 10 Blend Properties (Full Width) ---
        st.markdown("##### Resulting Blend Properties")
        prop_kpi_cols = st.columns(10)
        for i, prop_val in enumerate(selected_solution["blend_properties"]):
            with prop_kpi_cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 10px; padding: 0.5rem;">
                    <div class="metric-label" style="font-size: 0.7rem;">Property {i+1}</div>
                    <div class="metric-value" style="font-size: 1.1rem;">{prop_val:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

        # --- REPLACEMENT FOR THE "Cost Analysis" SECTION ---
        st.markdown("##### Performance Analysis")
        # Calculate baseline cost and quality score
        component_costs = [st.session_state.get(f"opt_c{i}_cost", 0.0) for i in range(5)]
        baseline_cost = sum(0.2 * cost for cost in component_costs)
        optimized_cost = selected_solution.get("optimized_cost", 0.0)
        quality_score = calculate_quality_score(selected_solution.get("error"))

        # Use more columns to make the cards smaller
        _, c1, c2, c3, c4, _ = st.columns([0.5, 1.5, 1.5, 1.5, 1.5, 0.5])
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Baseline Cost</div>
                <div class="metric-value">${baseline_cost:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Optimized Cost</div>
                <div class="metric-value">${optimized_cost:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            savings = baseline_cost - optimized_cost
            savings_color = "green" if savings >= 0 else "red"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Savings</div>
                <div class="metric-value" style="color:{savings_color};">${savings:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #8B4513; background: #FFF8E1;">
                <div class="metric-label">Quality Score</div>
                <div class="metric-value">{quality_score:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True) # Spacer
        
        # Expander for full results table
        with st.expander("Show Full Results Table"):
            table_data = []
            for i in range(5):
                row = {
                    "Composition": st.session_state.get(f"opt_c{i}_name", f"C{i+1}"),
                    "Fraction": selected_solution["component_fractions"][i],
                    "Unit Cost": st.session_state.get(f"opt_c{i}_cost", 0.0)
                }
                for j in range(1, 11):
                    row[f"Property {j}"] = st.session_state.get(f"opt_c{i}_prop{j}", 0.0)
                table_data.append(row)
            
            # Add blend row
            blend_row = {"Composition": "Optimized Blend", "Fraction": 1.0, "Unit Cost": selected_solution["optimized_cost"]}
            for i, prop in enumerate(selected_solution["blend_properties"]):
                blend_row[f"Property {i+1}"] = prop
            table_data.append(blend_row)
            
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # Pareto Plot and Save Section
        pareto_col, save_col = st.columns([2, 1])
        with pareto_col:
            st.markdown("##### Pareto Front: Cost vs. Error")
            pareto_df = pd.DataFrame({
                'Cost': [r['optimized_cost'] for r in results],
                'Error': [r['error'] for r in results],
                'Solution': [f'Sol {i+1}' for i in range(len(results))]
            })
            # --- FIX: Inverted the axes to show Error vs. Cost ---
            fig_pareto = px.scatter(
                pareto_df, x='Error', y='Cost', text='Solution', title="<b>Pareto Front: Error vs. Cost</b>"
            )
            fig_pareto.update_traces(textposition='top center', marker=dict(size=12, color='#8B4513'))
            st.plotly_chart(fig_pareto, use_container_width=True)

        with save_col:
            st.markdown("##### Save Result")
            st.text_input("Save as Blend Name", value=f"Optimized_Blend_{selected_idx+1}", key="opt_save_name")
            # --- REPLACEMENT FOR THE SAVE BUTTON LOGIC ---
            if st.button("💾 Save to Database", use_container_width=False):
                # Prepare DataFrame in the format expected by `add_blends`
                save_data = {}
                
                # 1. Add blend name
                save_name = st.session_state.get("opt_save_name", f"Optimized_Blend_{selected_idx+1}")
                save_data['blend_name'] = [save_name]
                
                # 2. Add component fractions and costs from UI
                for i in range(5):
                    save_data[f'Component{i+1}_fraction'] = selected_solution["component_fractions"][i]
                    for j in range(1, 11):
                        save_data[f'Component{i+1}_Property{j}'] = st.session_state.get(f"opt_c{i}_prop{j}", 0.0)
                    save_data[f'Component{i+1}_unit_cost'] = st.session_state.get(f"opt_c{i}_cost", 0.0)
                    
                # 3. Add the 10 final blend properties
                for i, prop_val in enumerate(selected_solution["blend_properties"], 1):
                    save_data[f'BlendProperty{i}'] = prop_val
                    
                # 4. Add the PreOpt (Baseline) and Optimized costs
                component_costs = [st.session_state.get(f"opt_c{i}_cost", 0.0) for i in range(5)]
                baseline_cost = sum(0.2 * cost for cost in component_costs)
                optimized_cost = selected_solution.get("optimized_cost", 0.0)
                
                save_data['PreOpt_Cost'] = [baseline_cost]
                save_data['Optimized_Cost'] = [optimized_cost]
                save_data['Quality_Score'] = [calculate_quality_score(selected_solution.get("error"))]
                
                save_df = pd.DataFrame(save_data)
                
                try:
                    result = add_blends(save_df)
                    log_activity("save_optimization", f"Saved optimized blend: {save_name}")
                    st.success(f"Successfully saved blend '{save_name}' to the database!")
                    get_all_blends_data.clear() # Clear cache for comparison tab
                except Exception as e:
                    st.error(f"Failed to save blend: {e}")
            
            # Placeholder for download button logic
            st.download_button("📥 Download All Solutions (CSV)", data="dummy_csv_data", file_name="optimization_results.csv", use_container_width=False)
            
    # --- Floating Help Button ---
    # (Using a different key to avoid conflict with other tabs)
    # --- FIX: Complete working version of the help button ---
    st.markdown("""
        <style>
            #help-toggle-optimizer { display: none; }
            #help-toggle-optimizer:checked ~ .help-panel-optimizer {
                opacity: 1; visibility: visible; transform: translateY(0);
            }
            .help-panel-optimizer {
                position:fixed; right:25px; bottom:100px; z-index:9998;
                width:520px; max-height:70vh; overflow-y:auto;
                background: linear-gradient(135deg, #FFFDF5 0%, #F8EAD9 100%);
                border:1px solid #CFB53B; border-radius:12px; padding:20px;
                box-shadow:0 14px 34px rgba(0,0,0,0.22);
                color:#4a2f1f; transform: translateY(12px); opacity:0;
                visibility:hidden; transition: all .22s ease-in-out;
            }
        </style>
        <input id="help-toggle-optimizer" type="checkbox" />
        <label for="help-toggle-optimizer" class="help-button">💬 Help</label>
        <div class="help-panel help-panel-optimizer"> <div class="head">
                <div class="title">How to Use the Optimizer</div>
                <label for="help-toggle-optimizer" class="help-close">Close</label>
            </div>
            <div class="help-body">
                <p><b>1. Define Goals:</b> Enter your desired target values for each of the 10 blend properties. Use the 'Fix Target' toggle for any property that must be met exactly.</p>
                <p><b>2. Select Components:</b> Choose up to 5 base components. You can load them from the registry to auto-fill their data or enter them manually.</p>
                <p><b>3. Configure & Run:</b> Decide if cost should be a factor in the optimization, then click 'Run Optimization'. A spinner will appear while the process runs.</p>
                <p><b>4. Analyze Results:</b> After completion, the best solution is shown by default. You can view other potential solutions from the dropdown. The results include optimal component fractions and the final blend properties.</p>
                <p><b>5. Save & Download:</b> Give your chosen solution a name and save it to the blends database for future use in the Comparison tab.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
##                                                   Blend Comparison Tab
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

@st.cache_data
def get_blend_property_ranges(db_path="eagleblend.db") -> dict:
    """Calculates the min and max for each BlendProperty across all blends."""
    ranges = {}
    with sqlite3.connect(db_path) as conn:
        for i in range(1, 11):
            prop_name = f"BlendProperty{i}"
            query = f"SELECT MIN({prop_name}), MAX({prop_name}) FROM blends WHERE {prop_name} IS NOT NULL"
            min_val, max_val = conn.execute(query).fetchone()
            ranges[prop_name] = (min_val if min_val is not None else 0, max_val if max_val is not None else 1)
    return ranges

with tabs[3]:
    st.subheader("📊 Blend Scenario Comparison")

    # --- Initial Data Loading ---
    all_blends_df = get_all_blends_data()
    property_ranges = get_blend_property_ranges()

    if all_blends_df.empty:
        st.warning("No blends found in the database. Please add blends in the 'Fuel Registry' tab to use this feature.")
    else:
        # --- Scenario Selection ---
        st.markdown("Select up to three blends from the registry to compare their properties and performance.")
        cols = st.columns(3)
        selected_blends = []
        blend_names = all_blends_df['blend_name'].tolist()

        for i, col in enumerate(cols):
            with col:
                choice = st.selectbox(
                    f"Select Blend for Scenario {i+1}",
                    options=["-"] + blend_names,
                    key=f"blend_select_{i}"
                )
                if choice != "-":
                    selected_blends.append(choice)
        
        # Filter the main dataframe to only include selected blends
        if selected_blends:
            #--- FIX: Filter duplicates to get only the most recent entry for each blend name ---
            filtered_df = all_blends_df[all_blends_df['blend_name'].isin(selected_blends)]
            comparison_df = filtered_df.sort_values('id', ascending=False).drop_duplicates(subset=['blend_name']).set_index('blend_name')
            
            # --- Information Cards ---
            st.markdown("---")
            # --- FIX: This new block creates a stable 3-column layout ---
            st.markdown("#### Selected Blend Overview")
            card_cols = st.columns(3)  # Create a fixed 3-column layout immediately
            for i, blend_name in enumerate(selected_blends):
                # Place each selected blend into its corresponding column
                with card_cols[i]:
                    blend_data = comparison_df.loc[blend_name]
                    #--- FIX: Use pd.isna() for a robust check of the timestamp value ---
                    created_val = blend_data.get('created_at')
                    created_at = pd.to_datetime(created_val).strftime('%Y-%m-%d') if not pd.isna(created_val) else 'N/A'
                    # Component Fractions
                    fractions_html = ""
                    for j in range(1, 6):
                        frac = blend_data.get(f"Component{j}_fraction", 0) * 100
                        if frac > 0:
                            fractions_html += f"<span style='font-weight: 500;'>C{j}:</span> {frac:.1f}% &nbsp; "
                    
                    # Blend Properties
                    properties_html = ""
                    for j in range(1, 11):
                        prop = blend_data.get(f"BlendProperty{j}")
                        if prop is not None:
                            properties_html += f"<span style='background: #f0f2f6; padding: 2px 5px; border-radius: 4px; margin-right: 4px; font-size: 0.8rem;'><b>P<sub>{j}</sub>:</b> {prop:.3f}</span>"

                    st.markdown(f"""
                    <div class="metric-card" style="padding: 1rem; height: 100%;">
                        <div class="metric-label" style="font-size: 1.1rem; text-align: left;">{blend_name}</div>
                        <div style="font-size: 0.8rem; text-align: left; color: #6c757d; margin-bottom: 10px;">Created: {created_at}</div>
                        <div style="font-size: 0.9rem; text-align: left; margin-bottom: 10px;">{fractions_html}</div>
                        <div style="text-align: left;">{properties_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # --- Charting Section ---
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            st.subheader("📈 Comparative Analysis")

            plot_cols = st.columns(2)
            with plot_cols[0]:
                # --- Plot 1: Sorted Bar Plot (Cost) ---
                # 1. Prepare data and sort it for clear visualization
                costs_data = []
                for name in selected_blends:
                    row = comparison_df.loc[name]
                    # Prioritize Optimized_Cost, then fall back to PreOpt_Cost
                    cost = row.get('Optimized_Cost') or row.get('PreOpt_Cost') or 0
                    if cost > 0: # Only include blends with a valid cost
                        costs_data.append({'Blend': name, 'Cost': cost})

                if costs_data:
                    cost_df = pd.DataFrame(costs_data)
                    cost_df = cost_df.sort_values(by='Cost', ascending=False)

                    # 2. Create the horizontal bar plot with Plotly Express
                    fig_cost = px.bar(
                        cost_df,
                        x='Cost',
                        y='Blend',
                        orientation='h',
                        text='Cost',
                        title="<b>Blend Cost Comparison</b>",
                        labels={'Cost': 'Cost ($ per unit)', 'Blend': ''} # Use an empty string to remove the y-axis title
                    )

                    # 3. Apply professional styling
                    fig_cost.update_traces(
                        marker_color='#8B4513', # Use a theme-consistent dark brown
                        marker_line_color='#4a2f1f',
                        marker_line_width=1.5,
                        texttemplate='$%{text:,.2f}', # Format text as currency
                        textposition='outside',
                        insidetextfont=dict(color='white')
                    )
                    fig_cost.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="#4a2f1f"),
                        uniformtext_minsize=8, 
                        uniformtext_mode='hide'
                    )
                    st.plotly_chart(fig_cost, use_container_width=True)
                else:
                    st.info("No cost data available for the selected blends to generate a comparison chart.")

            # --- This is the new, more robust radar chart block ---
            with plot_cols[1]:
                # --- Plot 2: Radar Chart (Blend Properties) ---
                categories = [f'P{i}' for i in range(1, 11)]
                radar_data_exists = False
                
                fig_radar = go.Figure()
                
                for name in selected_blends:
                    values = [comparison_df.loc[name].get(f'BlendProperty{i}', 0) for i in range(1, 11)]
                    # Check if there's any non-zero data to plot
                    if any(v > 0 for v in values):
                        radar_data_exists = True
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values, theta=categories, fill='toself', name=name
                    ))
                    
                # Only show the chart if there is data, otherwise show a warning
                if radar_data_exists:
                    fig_radar.update_layout(
                        title="<b>Blend Property Profile</b>",
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=True,
                        height=500,
                        margin=dict(l=80, r=80, t=100, b=80),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.warning("Radar Chart cannot be displayed. The selected blend(s) have no property data in the database.", icon="📊")
            # --- Plot 3 & 4 ---
            plot_cols2 = st.columns(2)
            with plot_cols2[0]:
                # --- Plot 3: Scatter Plot (Cost vs Quality) ---
                # 1. Prepare a self-contained DataFrame for this specific plot
                scatter_data = []
                for name in selected_blends:
                    row = comparison_df.loc[name]
                    cost = row.get('Optimized_Cost') or row.get('PreOpt_Cost') or 0
                    quality = row.get('Quality_Score', 0)

                    # Only include points that have valid data for both axes
                    if cost > 0 and quality > 0:
                        scatter_data.append({
                            'Blend': name, 
                            'Cost': cost, 
                            'Quality Score': quality
                        })

                # 2. Create the plot only if there is data to show
                if scatter_data:
                    scatter_df = pd.DataFrame(scatter_data)

                    fig_scatter = px.scatter(
                        scatter_df,
                        x='Cost',
                        y='Quality Score',
                        text='Blend',
                        labels={'Cost': 'Cost ($)', 'Quality Score': 'Quality Score'},
                        title="<b>Cost vs. Quality Frontier</b>"
                    )
                    fig_scatter.update_traces(
                        textposition='top center',
                        marker=dict(size=25, color='#8B4513', symbol='circle')
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Not enough cost and quality data to generate the Cost vs. Quality plot.")

            with plot_cols2[1]:
                # --- Plot 4: 100% Stacked Bar (Component Fractions) ---
                frac_data = comparison_df[[f'Component{i}_fraction' for i in range(1, 6)]].reset_index()
                frac_data_melted = frac_data.melt(id_vars='blend_name', var_name='Component', value_name='Fraction')
                
                fig_stacked = px.bar(
                    frac_data_melted, x='blend_name', y='Fraction', color='Component',
                    title="<b>Component Composition by Scenario</b>",
                    labels={'blend_name': 'Scenario'},
                    # --- FIX: Using a theme-consistent Yellow-Orange-Brown palette ---
                    # color_discrete_sequence=px.colors.sequential.YlOrBr_
                    # # --- FIX: Using Plotly's default palette for distinct colors (blue, red, green, etc.) ---
                    color_discrete_sequence=px.colors.qualitative.Plotly
                    # --- FIX: Using a qualitative palette for more distinct colors ---
                    # color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_stacked.update_layout(barmode='stack')
                st.plotly_chart(fig_stacked, use_container_width=True)

            # --- Plot 5: Composite Bar Chart ---
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            
            # --- FIX: Constrain selectbox width using columns ---
            s_col1, s_col2, s_col3 = st.columns([1, 2, 1])
            with s_col2:
                prop_idx = st.selectbox(
                    "Select Property to Visualize (Pj)",
                    options=list(range(1, 11)),
                    format_func=lambda x: f"Property {x}",
                    key="composite_prop_select",
                    label_visibility="collapsed" # Hides the label to make it cleaner
                )

            comp_prop_name = f'Component{{}}_Property{prop_idx}'
            blend_prop_name = f'BlendProperty{prop_idx}'
            
            chart_data = []
            for name in selected_blends:
                for i in range(1, 6): # Components C1-C5
                    chart_data.append({
                        'Scenario': name,
                        'Composition': f'C{i}',
                        'Value': comparison_df.loc[name].get(comp_prop_name.format(i), 0)
                    })
                # Blend Property
                chart_data.append({
                    'Scenario': name,
                    'Composition': 'Blend',
                    'Value': comparison_df.loc[name].get(blend_prop_name, 0)
                })

            composite_df = pd.DataFrame(chart_data)
            
            fig_composite = px.line(
                composite_df, x='Composition', y='Value', color='Scenario',
                markers=True, title=f"<b>Comparative Analysis for Property {prop_idx}</b>",
                labels={'Composition': 'Composition (C1-C5 & Blend)', 'Value': f'Property {prop_idx} Value'}
            )
            st.plotly_chart(fig_composite, use_container_width=True)

    # --- ADD: Floating Help Button for Blend Comparison ---
    st.markdown("""
        <style>
            #help-toggle-comparison { display: none; }
            #help-toggle-comparison:checked ~ .help-panel-comparison {
                opacity: 1; visibility: visible; transform: translateY(0);
            }
            .help-panel-comparison {
                position:fixed; right:25px; bottom:100px; z-index:9998;
                width:520px; max-height:70vh; overflow-y:auto;
                background: linear-gradient(135deg, #FFFDF5 0%, #F8EAD9 100%);
                border:1px solid #CFB53B; border-radius:12px; padding:20px;
                box-shadow:0 14px 34px rgba(0,0,0,0.22);
                color:#4a2f1f; transform: translateY(12px); opacity:0;
                visibility:hidden; transition: all .22s ease-in-out;
            }
        </style>
        <input id="help-toggle-comparison" type="checkbox" />
        <label for="help-toggle-comparison" class="help-button">💬 Help</label>
        <div class="help-panel help-panel-comparison">
            <div class="head">
                <div class="title">Using the Blend Comparison Tool</div>
                <label for="help-toggle-comparison" class="help-close">Close</label>
            </div>
            <div class="help-body">
                <p>This tab allows you to perform a side-by-side analysis of up to three saved blends.</p>
                <p><b>1. Select Scenarios:</b> Use the three dropdown menus at the top to select the saved blends you wish to compare.</p>
                <p><b>2. Review Overviews:</b> Key information for each selected blend, including its composition and final properties, will be displayed in summary cards.</p>
                <p><b>3. Analyze Charts:</b> The charts provide a deep dive into how the blends compare on cost, property profiles, quality, and composition.</p>
                <p><b>4. Export:</b> Click the 'Export to PDF' button to generate a downloadable report containing all the charts and data for your selected comparison.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------------------------------------------------
#                                                   Fuel Registry Tab
# ---------------------------------------------------------------------------------------------------------------------------------------------


def load_data(table_name: str, db_path="eagleblend.db") -> pd.DataFrame:
    """Loads data from a specified table in the database."""
    try:
        conn = sqlite3.connect(db_path)
        # Assuming each table has a unique ID column as the first column
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Failed to load data from table '{table_name}': {e}")
        return pd.DataFrame()

def delete_records(table_name: str, ids_to_delete: list, id_column: str, db_path="eagleblend.db"):
    """Deletes records from a table based on a list of IDs."""
    if not ids_to_delete:
        return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        placeholders = ','.join('?' for _ in ids_to_delete)
        query = f"DELETE FROM {table_name} WHERE {id_column} IN ({placeholders})"
        cur.execute(query, ids_to_delete)
        conn.commit()
    finally:
        conn.close()

@st.cache_data
def get_template(file_path):
    """Loads a template file into bytes for downloading."""
    with open(file_path, 'rb') as f:

        return f.read()

with tabs[4]:
    st.subheader("📚 Fuel Registry")
    st.write("Manage fuel components and blends. Add new entries manually, upload in batches, or download templates.")
    
    # --- State Initialization ---
    if 'components' not in st.session_state:
        st.session_state.components = load_data('components')
    if 'blends' not in st.session_state:
        st.session_state.blends = load_data('blends')

    # --- Section 1: Data Management (Uploads & Manual Entry) ---
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### ➕ Add Components")
            
            with st.expander("Add a Single Component Manually"):
                with st.form("new_component_form", clear_on_submit=True):
                    component_name = st.text_input("Component Name", placeholder="e.g., Reformate")
                    
                    # --- FIX: Add Component Type dropdown ---
                    component_type = st.selectbox("Component Type", options=["-- Select a Type --", 1, 2, 3, 4, 5])
                    
                    c_cols = st.columns(2)
                    # Gather all property inputs
                    property1 = c_cols[0].number_input("Property 1", value=0.0, step=0.1, format="%.4f")
                    property2 = c_cols[1].number_input("Property 2", value=0.0, step=0.1, format="%.4f")
                    property3 = c_cols[0].number_input("Property 3", value=0.0, step=0.1, format="%.4f")
                    property4 = c_cols[1].number_input("Property 4", value=0.0, step=0.1, format="%.4f")
                    property5 = c_cols[0].number_input("Property 5", value=0.0, step=0.1, format="%.4f")
                    property6 = c_cols[1].number_input("Property 6", value=0.0, step=0.1, format="%.4f")
                    property7 = c_cols[0].number_input("Property 7", value=0.0, step=0.1, format="%.4f")
                    property8 = c_cols[1].number_input("Property 8", value=0.0, step=0.1, format="%.4f")
                    property9 = c_cols[0].number_input("Property 9", value=0.0, step=0.1, format="%.4f")
                    property10 = c_cols[1].number_input("Property 10", value=0.0, step=0.1, format="%.4f")
                    unit_cost = c_cols[0].number_input("Unit Cost", value=0.0, step=0.1, format="%.2f")

                    if st.form_submit_button("💾 Save Component", use_container_width=True):
                        # --- FIX: Add validation for component type ---
                        if not component_name.strip():
                            st.warning("Component Name cannot be empty.")
                        elif component_type == "-- Select a Type --":
                            st.warning("Please select a Component Type.")
                        else:
                            # --- FIX: Include component_type in the data to be saved ---
                            new_component_data = {
                                "component_name": component_name,
                                "component_type": component_type,
                                "property1": property1, "property2": property2,
                                "property3": property3, "property4": property4,
                                "property5": property5, "property6": property6,
                                "property7": property7, "property8": property8,
                                "property9": property9, "property10": property10,
                                "unit_cost": unit_cost
                            }
                            new_component_df = pd.DataFrame([new_component_data])
                            rows_added = add_components(new_component_df)
                            if rows_added > 0:
                                st.success(f"Component '{component_name}' added successfully!")
                                if 'components' in st.session_state:
                                    del st.session_state.components
                                st.rerun()
            
            # Batch upload for components
            st.markdown("---")
            st.markdown("**Batch Upload Components**")
            uploaded_components = st.file_uploader(
                "Upload Components CSV", type=['csv'], key="components_uploader",
                help="Upload a CSV file with component properties."
            )
            if uploaded_components:
                try:
                    df = pd.read_csv(uploaded_components)
                    rows_added = add_components(df)
                    st.success(f"Successfully added {rows_added} new components to the registry!")
                    del st.session_state.components # Force reload
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing file: {e}")

            st.download_button(
                label="📥 Download Component Template",
                data=get_template('assets/components_template.csv'),
                file_name='components_template.csv',
                mime='text/csv',
                use_container_width=True
            )

    with col2:
        with st.container(border=True):
            st.markdown("#### 🧬 Add Blends")
            st.info("Upload blend compositions via CSV. Manual entry is not supported for blends.", icon="ℹ️")

            # Batch upload for blends
            uploaded_blends = st.file_uploader(
                "Upload Blends CSV", type=['csv'], key="blends_uploader",
                help="Upload a CSV file defining blend recipes."
            )
            if uploaded_blends:
                try:
                    df = pd.read_csv(uploaded_blends)
                    rows_added = add_blends(df) # Assumes you have an add_blends function
                    st.success(f"Successfully added {rows_added} new blends to the registry!")
                    del st.session_state.blends # Force reload
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing file: {e}")

            st.download_button(
                label="📥 Download Blend Template",
                data=get_template('assets/blends_template.csv'),
                file_name='blends_template.csv',
                mime='text/csv',
                use_container_width=True
            )
    
    st.divider()

    # --- Section 2: Data Display & Deletion ---
    st.markdown("#### 🔍 View & Manage Registry Data")
    
    view_col1, view_col2 = st.columns([1, 2])
    
    with view_col1:
        table_to_show = st.selectbox(
            "Select Table to View",
            ("Components", "Blends"),
            label_visibility="collapsed"
        )

    with view_col2:
        search_query = st.text_input(
            "Search Table",
            placeholder=f"Type to search in {table_to_show}...",
            label_visibility="collapsed"
        )
        
    # Determine which DataFrame to use
    if table_to_show == "Components":
        df_display = st.session_state.components.copy()
        id_column = "id" # Change if your ID column is named differently
    else:
        df_display = st.session_state.blends.copy()
        id_column = "id" # Change if your ID column is named differently

    # Apply search filter if query is provided
    if search_query:
        # A simple search across all columns
        df_display = df_display[df_display.apply(
            lambda row: row.astype(str).str.contains(search_query, case=False).any(),
            axis=1
        )]
    
    if df_display.empty:
        st.warning(f"No {table_to_show.lower()} found matching your criteria.")
    else:
        # Add a "Select" column for deletion
        df_display.insert(0, "Select", False)
        
        # Use data_editor to make the checkboxes interactive
        edited_df = st.data_editor(
            df_display,
            hide_index=True,
            use_container_width=True,
            disabled=df_display.columns.drop("Select"), # Make all columns except "Select" read-only
            key=f"editor_{table_to_show}"
        )
        
        selected_rows = edited_df[edited_df["Select"]]
        
        if not selected_rows.empty:
            # --- FIX: Reverted to the full-width button as requested ---
            if st.button(f"❌ Delete Selected {table_to_show} ({len(selected_rows)})", use_container_width=False, type="primary"):
                ids_to_del = selected_rows['id'].tolist()
                delete_records(table_to_show.lower(), ids_to_del, 'id')
                st.success(f"Deleted {len(ids_to_del)} records from {table_to_show}.")
                
                # Clear the relevant cache to reflect the deletion
                if table_to_show == "Components":
                    if 'components' in st.session_state:
                        del st.session_state.components
                else:
                    if 'blends' in st.session_state:
                        del st.session_state.blends
                st.rerun()
                # st.rerun()

    # --- ADD: Floating Help Button for Fuel Registry ---
    st.markdown("""
        <style>
            #help-toggle-registry { display: none; }
            #help-toggle-registry:checked ~ .help-panel-registry {
                opacity: 1; visibility: visible; transform: translateY(0);
            }
            .help-panel-registry {
                position:fixed; right:25px; bottom:100px; z-index:9998;
                width:520px; max-height:70vh; overflow-y:auto;
                background: linear-gradient(135deg, #FFFDF5 0%, #F8EAD9 100%);
                border:1px solid #CFB53B; border-radius:12px; padding:20px;
                box-shadow:0 14px 34px rgba(0,0,0,0.22);
                color:#4a2f1f; transform: translateY(12px); opacity:0;
                visibility:hidden; transition: all .22s ease-in-out;
            }
        </style>
        <input id="help-toggle-registry" type="checkbox" />
        <label for="help-toggle-registry" class="help-button">💬 Help</label>
        <div class="help-panel help-panel-registry">
            <div class="head">
                <div class="title">Using the Fuel Registry</div>
                <label for="help-toggle-registry" class="help-close">Close</label>
            </div>
            <div class="help-body">
                <p>This tab is your central database for managing all blend components and saved blends.</p>
                <p><b>1. Add Components/Blends:</b> You can add a single component manually using the form or upload a CSV file for batch additions of components or blends. Download the templates to ensure your file format is correct.</p>
                <p><b>2. View & Manage Data:</b> Use the dropdown to switch between viewing 'Components' and 'Blends'. The table shows all saved records.</p>
                <p><b>3. Search & Delete:</b> Use the search bar to filter the table. To delete records, check the 'Select' box next to the desired rows and click the 'Delete Selected' button that appears.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    

# ----------------------------------------------------------------------------------------------------------------------------------------------
#                                               Model Insights Tab
# ----------------------------------------------------------------------------------------------------------------------------------------------
with tabs[5]:

    model_metrics = last_model[
        [f"BlendProperty{i}_Score" for i in range(1, 11)]
    ]

    # --- UI Rendering Starts Here ---

    # Inject CSS for consistent styling with the rest of the app
    st.markdown("""
    <style>
    /* Metric card styles */
    .metric-card {
        background: linear-gradient(180deg, #FFF8E1 0%, #FFF6EA 100%);
        border: 1px solid #E3C77A;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        color: #654321;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 14px;
        font-weight: 700;
        color: #8B4513;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 900;
        color: #4a2f1f;
    }
    /* Floating help button and panel styles */
    #help-toggle{display:none;}
    .help-button{
        position:fixed; right:25px; bottom:25px; z-index:9999;
        background:#8B4513; color:#FFD700; padding:16px 22px; font-size:17px;
        border-radius:18px; font-weight:900; box-shadow:0 8px 22px rgba(0,0,0,0.2); cursor:pointer;
        border:0;
    }
    .help-panel{
        position:fixed; right:25px; bottom:100px; z-index:9998;
        width:520px; max-height:70vh; overflow-y:auto;
        background: linear-gradient(135deg, #FFFDF5 0%, #F8EAD9 100%);
        border:1px solid #CFB53B; border-radius:12px; padding:20px; box-shadow:0 14px 34px rgba(0,0,0,0.22);
        color:#4a2f1f; transform: translateY(12px); opacity:0; visibility:hidden; transition: all .22s ease-in-out;
    }
    #help-toggle:checked + label.help-button + .help-panel{
        opacity:1; visibility:visible; transform: translateY(0);
    }
    .help-panel .head{display:flex; justify-content:space-between; align-items:center; margin-bottom:12px}
    .help-panel .title{font-weight:900; color:#654321; font-size:16px}
    .help-close{background:#8B4513; color:#FFD700; padding:6px 10px; border-radius:8px; cursor:pointer; font-weight:800}
    .help-body{font-size:14.5px; color:#4a2f1f; line-height:1.5}
    .help-body b {color: #654321;}
    </style>
    """, unsafe_allow_html=True)

    # # --- Floating "How to Use" Button and Panel ---
    # st.markdown("""
    # <input id="help-toggle" type="checkbox" />
    # <label for="help-toggle" class="help-button">💬 Help</label>

    # <div class="help-panel" aria-hidden="true">
    #     <div class="head">
    #         <div class="title">Interpreting Model Insights</div>
    #         <label for="help-toggle" class="help-close">Close</label>
    #     </div>
    #     <div class="help-body">
    #         <p><b>KPI Cards:</b> These four cards give you a quick summary of the model's overall health.</p>
    #         <ul>
    #             <li><b>Overall R² Score:</b> Think of this as the model's accuracy grade. A score of 92.4% means the model's predictions are highly accurate.</li>
    #             <li><b>MSE (Mean Squared Error):</b> This measures the average size of the model's mistakes. A smaller number is better.</li>
    #             <li><b>MAPE (Mean Absolute % Error):</b> This tells you the average error in percentage terms. A value of 0.112 means predictions are off by about 11.2% on average.</li>
    #         </ul>
    #         <p><b>R² Score by Blend Property Chart:</b> This chart shows how well the model predicts each specific property.</p>
    #         <p>A <b>longer bar</b> means the model is very good at predicting that property. A <b>shorter bar</b> indicates a property that is harder for the model to predict accurately. This helps you trust predictions for some properties more than others.</p>
    #     </div>
    # </div>
    # """, unsafe_allow_html=True)

        # --- FIX: Complete working version of the help button ---
# --- FIX: Complete working version of the help button ---
    st.markdown("""
        <style>
            /* Styles for the help panel and button */
            #help-toggle-insights { display: none; }
            #help-toggle-insights:checked ~ .help-panel-insights {
                opacity: 1; visibility: visible; transform: translateY(0);
            }
            .help-panel-insights {
                position:fixed; right:25px; bottom:100px; z-index:9998;
                width:520px; max-height:70vh; overflow-y:auto;
                background: linear-gradient(135deg, #FFFDF5 0%, #F8EAD9 100%);
                border:1px solid #CFB53B; border-radius:12px; padding:20px;
                box-shadow:0 14px 34px rgba(0,0,0,0.22);
                color:#4a2f1f; transform: translateY(12px); opacity:0;
                visibility:hidden; transition: all .22s ease-in-out;
            }
        </style>
        <input id="help-toggle-insights" type="checkbox" />
        <label for="help-toggle-insights" class="help-button">💬 Help</label>
        <div class="help-panel help-panel-insights">
            <div class="head">
                <div class="title">Interpreting Model Insights</div>
                <label for="help-toggle-insights" class="help-close">Close</label>
            </div>
            <div class="help-body">
                <p><b>KPI Cards:</b> These cards give a quick summary of the model's health. <b>R² Score</b> is its accuracy grade, while <b>MSE</b> and <b>MAPE</b> measure the average size of its errors.</p>
                <p><b>R² Score by Blend Property Chart:</b> This chart shows how well the model predicts each specific property. A longer bar means the model is very good at predicting that property.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- Main Title ---
    st.markdown('<h2 style="color:#4a2f1f; font-size:1.75rem;">🧠 Model Insights</h2>', unsafe_allow_html=True)

    # --- Fetch Model Data ---
    latest_model = get_model()
    model_name = latest_model.get("model_name", "N/A")
    r2_score = f'{latest_model.get("R2_Score", 0) * 100:.1f}%'
    mse = f'{latest_model.get("MSE", 0):.3f}'
    mape = f'{latest_model.get("MAPE", 0):.3f}'

    # --- KPI Cards Section ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Name</div>
                <div class="metric-value" style="font-size: 1.2rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{model_name}</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall R² Score</div>
                <div class="metric-value">{r2_score}</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Squared Error</div>
                <div class="metric-value">{mse}</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Absolute % Error</div>
                <div class="metric-value">{mape}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True) # Spacer

    # --- R2 Score by Property Chart ---
    st.markdown('<h3 style="color:#4a2f1f; font-size:1.5rem;">R² Score by Blend Property</h3>', unsafe_allow_html=True)

    # Create the horizontal bar chart
    fig_r2 = go.Figure()

    fig_r2.add_trace(go.Bar(
        y=model_metrics.index,
        x=model_metrics.values,
        orientation='h',
        marker=dict(
            color=model_metrics.values,
            colorscale='YlOrBr',
            colorbar=dict(title="R² Score", tickfont=dict(color="#4a2f1f")),
        ),
        text=[f'{val:.2f}' for val in model_metrics.values],
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color='#4a2f1f', size=12, family='Arial, sans-serif', weight='bold')
    ))

    # This corrected block resolves the ValueError
    fig_r2.update_layout(
        xaxis_title="R² Score (Higher is Better)",
        yaxis_title="Blend Property",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=20, b=50),
        font=dict(
            family="Segoe UI, Arial, sans-serif",
            size=12,
            color="#4a2f1f"
        ),
        yaxis=dict(
            tickfont=dict(size=12, weight='bold'),
            automargin=True,
            # FIX: The title font styling is now correctly nested here
            title_font=dict(size=14)
        ),
        xaxis=dict(
            gridcolor="rgba(139, 69, 19, 0.2)",
            zerolinecolor="rgba(139, 69, 19, 0.3)",
            # FIX: The title font styling is now correctly nested here
            title_font=dict(size=14)
        )
    )

    st.plotly_chart(fig_r2, use_container_width=True)




#     st.markdown("""
#     <style>
#     /* Consistent chart styling */
#     .stPlotlyChart {
#         border-radius: 10px;
#         background: white;
#         padding: 15px;
#         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
#         margin-bottom: 25px;
#     }
    
#     /* Better select widget alignment */
#     .stSelectbox > div {
#         margin-bottom: -15px;
#     }
    
#     /* Color scale adjustments */
#     .plotly .colorbar {
#         padding: 10px !important;
#     }
#     </style>
# """, unsafe_allow_html=True)