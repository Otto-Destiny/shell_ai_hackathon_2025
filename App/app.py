# app.py
import streamlit as st
import pandas as pd
# from blend_logic import run_dummy_prediction

# ---------------------- Page Config ----------------------
st.set_page_config(
    layout="wide",
    page_title="Eagle Blend Optimizer",
    page_icon="🦅",
    initial_sidebar_state="expanded",
    #  theme={
    #     "primaryColor": "#ff4b4b",
    #     "backgroundColor": "#f0f2f6",  # This changes the main content area
    #     "secondaryBackgroundColor": "#ffffff",  # This changes other elements
    #     "textColor": "#31333F",
    # }
    
)

# ---------------------- Custom Styling ----------------------


# st.markdown("""
#     <style>
#     /* Set full background */
#     body {
#         background-color: #f3eaff;
#     }

#     /* Main content container */
#     .main {
#         background-color: #f3eaff;
#     }
            
#     .stApp {
#         background-color: #6c74e0;
#     }
            
#     /* Style the tabs */
#     .stTabs [data-baseweb="tab-list"] {
#         display: flex;
#         justify-content: space-between;
#         flex-wrap: wrap;
#         background-color: #ffffffcc;
#         border-radius: 12px;
#         padding: 10px;
#         margin-bottom: 20px;
#     }

#     .stTabs [data-baseweb="tab"] {
#         background-color: #ede1fa;
#         border-radius: 10px;
#         padding: 8px 16px;
#         margin: 6px;
#         font-weight: 600;
#         color: #4b0082;
#         flex-grow: 1;
#         text-align: center;
#     }

#     .stTabs [data-baseweb="tab"]:hover {
#         background-color: #d1b3ff;
#         color: #3a0066;
#         cursor: pointer;
#     }

#     .stTabs [aria-selected="true"] {
#         background-color: #cda9ff;
#         color: #000000;
#         box-shadow: 0 0 0 2px #b48cf1 inset;
#     }

#     /* Responsive tweak for mobile */
#     @media only screen and (max-width: 768px) {
#         .stTabs [data-baseweb="tab-list"] {
#             flex-direction: column;
#             align-items: stretch;
#         }

#         .stTabs [data-baseweb="tab"] {
#             margin: 4px 0;
#             flex-grow: unset;
#         }
#     }

#     /* Style section headings and sliders for contrast */
#     .block-container {
#         padding-top: 2rem;
#     }
#     .stSlider, .stSelectbox, .stButton {
#         background-color: #ffffffcc;
#         padding: 1rem;
#         border-radius: 10px;
#         margin-top: 0.5rem;
#     }
#     </style>
# """, unsafe_allow_html=True)



# # ---------------------- App Header ----------------------
# st.markdown("""
#     <h1 style='text-align: center; color: #ff6600;'>🦅 Eagle Blend Optimizer</h1>
#     <h4 style='text-align: center; color: #666;'>AI-Powered Fuel Blend Property Prediction & Optimization</h4>
#     <br>
# """, unsafe_allow_html=True)


# st.markdown("""
#     <style>
#     /* Main app background */
#     .stApp {
#         background-color: #f8f9fa;
#     }
    
#     /* Header styling */
#     .header {
#         background: linear-gradient(135deg, #1d3b58 0%, #2c5282 100%);
#         color: white;
#         padding: 2rem 1rem;
#         margin-bottom: 2rem;
#         border-radius: 0 0 15px 15px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
    
#     /* Card styling for metrics */
#     .metric-card {
#         background: white;
#         border-radius: 10px;
#         padding: 1.5rem;
#         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
#         height: 100%;
#         transition: transform 0.3s ease, box-shadow 0.3s ease;
#     }
    
#     .metric-card:hover {
#         transform: translateY(-5px);
#         box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
#     }
    
#     /* Tab styling */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 8px;
#         padding: 8px;
#         background-color: transparent;
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         background: white;
#         border-radius: 8px;
#         padding: 10px 16px;
#         margin: 0;
#         font-weight: 600;
#         color: #495057;
#         border: 1px solid #dee2e6;
#         transition: all 0.3s ease;
#     }
    
#     .stTabs [data-baseweb="tab"]:hover {
#         background-color: #e9ecef;
#         color: #1d3b58;
#     }
    
#     .stTabs [aria-selected="true"] {
#         background-color: #1d3b58;
#         color: white;
#         border-color: #1d3b58;
#     }
    
#     /* Button styling */
#     .stButton>button {
#         background-color: #1d3b58;
#         color: white;
#         border-radius: 8px;
#         padding: 0.5rem 1rem;
#         transition: all 0.3s ease;
#     }
    
#     .stButton>button:hover {
#         background-color: #2c5282;
#         color: white;
#     }
    
#     /* Dataframe styling */
#     .stDataFrame {
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
#     }
    
#     /* Section headers */
#     .st-emotion-cache-16txtl3 {
#         padding-top: 1rem;
#     }
    
#     /* Custom hr style */
#     .custom-divider {
#         border: 0;
#         height: 1px;
#         background: linear-gradient(90deg, transparent, #dee2e6, transparent);
#         margin: 2rem 0;
#     }
#     </style>
# """, unsafe_allow_html=True)



# ---------------------- Tabs ----------------------
tabs = st.tabs([
    "📊 Dashboard",
    "🎛️ Blend Designer",
    "📈 Property Analyzer",
    "⚙️ Optimization Engine",
    "📤 Batch Processing",
    "🧠 Model Insights"
])

# ---------------------- Dashboard Tab ----------------------
with tabs[0]:
    st.subheader("📊 Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Accuracy", "94.7%", "R² Score")
    col2.metric("Predictions Made", "12,847", "Today")
    col3.metric("Optimizations", "156", "This Week")
    col4.metric("Cost Savings", "$2.4M", "Estimated Annual")

    st.markdown("---")
    st.subheader("📊 Current Blend Properties")
    blend_props = {
        "Property 1": 0.847,
        "Property 2": 0.623,
        "Property 3": 0.734,
        "Property 4": 0.912,
        "Property 5": 0.456
    }
    st.dataframe(pd.DataFrame(blend_props.items(), columns=["Property", "Value"]))

# ---------------------- Blend Designer Tab ----------------------
# ---------------------- Blend Designer Tab ----------------------
with tabs[1]:
    st.subheader("🎛️ Blend Designer")

    # Property selection
    selected_property_idx = st.selectbox(
        "Select Property to Predict in Blend", 
        [f"Property {i+1}" for i in range(10)]
    )
    
    st.markdown("#### 🔢 Component Weights and Selected Property Values")
    
    weights, props = [], []
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### ⚖️ Component Weights")
        for i in range(5):
            weight = st.number_input(
                f"Weight for Component {i+1}", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.2, 
                step=0.01, 
                key=f"w_{i}"
            )
            weights.append(weight)

    with col2:
        st.markdown(f"##### 🧪 {selected_property_idx} Values for Each Component")
        for i in range(5):
            prop = st.number_input(
                f"Component {i+1} - {selected_property_idx}", 
                min_value=-6.0, 
                max_value=6.0, 
                value=0.5, 
                step=0.01, 
                key=f"p_{i}"
            )
            props.append(prop)

    st.markdown("---")
    if st.button("⚙️ Predict Blended Property"):
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 0.01:
            st.warning("⚠️ The total of weights must be **1.0**.")
        else:
            blended_value = sum(w * p for w, p in zip(weights, props))
            st.success(
                f"✅ Blended value for **{selected_property_idx}**: **{blended_value:.4f}**"
            )


# ---------------------- Property Analyzer ----------------------
with tabs[2]:
    st.subheader("📈 Property Analyzer")
    st.markdown("Advanced property/component relationships coming soon.")

# ---------------------- Optimization Engine ----------------------
with tabs[3]:
    st.subheader("⚙️ Optimization Engine")
    st.markdown("Placeholder for future optimization logic.")

# ---------------------- Batch Processing ----------------------
with tabs[4]:
    st.subheader("📤 Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully")
        st.dataframe(df.head())

        if st.button("⚙️ Run Batch Prediction"):
            result_df = df.copy()
            result_df["Predicted_Property"] = df.apply(
                lambda row: run_dummy_prediction(row.values[:5], row.values[5:10]), axis=1
            )
            st.success("Batch prediction completed")
            st.dataframe(result_df.head())
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "prediction_results.csv", "text/csv")

# ---------------------- Model Insights ----------------------
with tabs[5]:
    st.subheader("🧠 Model Insights")
    st.markdown("Future section for model explanations, SHAP values, etc.")
