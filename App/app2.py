import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot  as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
# from blend_logic import run_dummy_prediction

##---- fucntions ------
import pandas as pd
import streamlit as st

# Load fuel data from CSV (create this file if it doesn't exist)
FUEL_CSV_PATH = "fuel_properties.csv"

def load_fuel_data():
    """Load fuel data from CSV or create default if not exists"""
    try:
        df = pd.read_csv(FUEL_CSV_PATH, index_col=0)
        return df.to_dict('index')
    except FileNotFoundError:
        # Create default fuel properties if file doesn't exist
        default_fuels = {
            "Gasoline": {f"Property{i+1}": round(0.7 + (i*0.02), 1) for i in range(10)},
            "Diesel": {f"Property{i+1}": round(0.8 + (i*0.02), 1) for i in range(10)},
            "Ethanol": {f"Property{i+1}": round(0.75 + (i*0.02), 1) for i in range(10)},
            "Biodiesel": {f"Property{i+1}": round(0.85 + (i*0.02), 1) for i in range(10)},
            "Jet Fuel": {f"Property{i+1}": round(0.78 + (i*0.02), 1) for i in range(10)}
        }
        pd.DataFrame(default_fuels).T.to_csv(FUEL_CSV_PATH)
        return default_fuels

# Initialize or load fuel data
if 'FUEL_PROPERTIES' not in st.session_state:
    st.session_state.FUEL_PROPERTIES = load_fuel_data()

def save_fuel_data():
    """Save current fuel data to CSV"""
    pd.DataFrame(st.session_state.FUEL_PROPERTIES).T.to_csv(FUEL_CSV_PATH)

# FUEL_PROPERTIES = st.session_state.FUEL_PROPERTIES

# ---------------------- Page Config ----------------------
st.set_page_config(
    layout="wide",
    page_title="Eagle Blend Optimizer",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

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
            

    /* Consistent chart styling */
    .stPlotlyChart {
        border-radius: 10px;
        background: white;
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

    

            
    
    /* Color scale adjustments */
    .plotly .colorbar {
        padding: 10px !important;
            color: #654321 !important;
    }   

    </style>
""", unsafe_allow_html=True)

# ---------------------- App Header ----------------------
st.markdown("""
    <div class="header">
        <h1 style='text-align: center; margin-bottom: 0.5rem;'>🦅 Eagle Blend Optimizer</h1>
        <h4 style='text-align: center; font-weight: 400; margin-top: 0;'>
            AI-Powered Fuel Blend Property Prediction & Optimization
        </h4>
    </div>
""", unsafe_allow_html=True)
#------ universal variables
 

# ---------------------- Tabs ----------------------
tabs = st.tabs([
    "📊 Dashboard",
    "🎛️ Blend Designer",
    "📤 Nothing For Now",
    "⚙️ Optimization Engine",
    "📚 Fuel Registry",
    "🧠 Model Insights"
])

# ---------------------- Dashboard Tab ----------------------

with tabs[0]:
    st.subheader("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Model Accuracy</div>
                <div class="metric-value">94.7%</div>
                <div class="metric-delta">R² Score</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Predictions Made</div>
                <div class="metric-value">12,847</div>
                <div class="metric-delta">Today</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Optimizations</div>
                <div class="metric-value">156</div>
                <div class="metric-delta">This Week</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Cost Savings</div>
                <div class="metric-value">$2.4M</div>
                <div class="metric-delta">Estimated Annual</div>
            </div>
        """, unsafe_allow_html=True)



    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    


    st.subheader("Current Blend Properties")
    blend_props = {
        "Property 1": 0.847,
        "Property 2": 0.623,
        "Property 3": 0.734,
        "Property 4": 0.912,
        "Property 5": 0.456,
        "Property 6": -1.234,
    }
    
    # Enhanced dataframe display
    df = pd.DataFrame(blend_props.items(), columns=["Property", "Value"])
    # st.dataframe(
    #     df.style
    #     .background_gradient(cmap="YlOrBr", subset=["Value"])
    #     .format({"Value": "{:.3f}"}),
    #     use_container_width=True
    # )

    st.markdown('<div class="table-container"><div class="table-inner">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)




with tabs[1]:
    col_header = st.columns([0.8, 0.2])
    with col_header[0]:
        st.subheader("🎛️ Blend Designer")
    with col_header[1]:
        batch_blend = st.checkbox("Batch Blend Mode", value=False,
                                help="Switch between manual input and predefined fuel selection",
                                key="batch_blend_mode")

    # Initialize session state
    if 'show_visualization' not in st.session_state:
        st.session_state.show_visualization = False
    if 'blended_value' not in st.session_state:
        st.session_state.blended_value = None
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = "Property1"

    # Batch mode file upload
    if batch_blend:
        st.subheader("📤 Batch Processing")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="Batch_upload")
        weights = [0.1, 0.2, 0.25, 0.15, 0.3]  # Default weights for batch mode
        
        if not uploaded_file:
            st.warning("Please upload a CSV file for batch processing")
            data_input = None
        else:
            try:
                data_input = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully")
                st.dataframe(data_input.head())
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                data_input = None
    else:
        # Regular mode
        data_input = None
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
            st.markdown("##### Fuel Selection")
            for i in range(5):
                fuel = st.selectbox(
                    f"Component {i+1} Fuel Type",
                    options=list(st.session_state.FUEL_PROPERTIES.keys()),
                    key=f"fuel_{i}"
                )
                props.append(st.session_state.FUEL_PROPERTIES[fuel])

    if st.button("⚙️ Predict Blended Property", key="predict_btn"):
        if batch_blend:
            if data_input is None:
                st.error("⚠️ Please upload a valid CSV file first!")
                st.session_state.show_visualization = False
            else:
                st.session_state.show_visualization = True
        else:
            if abs(sum(weights) - 1.0) > 0.01:
                st.warning("⚠️ The total of weights must be **1.0**.")
                st.session_state.show_visualization = False
            else:
                st.session_state.show_visualization = True
                
        if st.session_state.show_visualization:
            # Show calculation details
            st.subheader("Blend Components Data")

            if not batch_blend:
                weights_data = {f"Component{i+1}_fraction": weights[i] for i in range(len(weights))}
                props_data = {f"Component{i+1}_{j}": props[i][j] for j in props[i].keys() for i in range(len(props))}
                combined = {**weights_data, **props_data}
                data_input = pd.DataFrame([combined])

            st.write("Properties:", data_input)

    # Show visualization only if prediction was made
    if st.session_state.show_visualization:
        if not batch_blend:
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            st.subheader("Blend Visualization")
            
            components = [f"Component {i+1}" for i in range(5)]
            
            # 1. Weight Distribution Pie Chart
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.pie(
                    names=components,
                    values=weights,
                    title="Weight Distribution",
                    color_discrete_sequence=['#8B4513', '#CFB53B', '#654321'],
                    hole=0.4
                )
                fig1.update_layout(
                    margin=dict(t=50, b=10),
                    showlegend=False
                )
                fig1.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#ffffff', width=1))
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            # 2. Property Comparison Bar Chart
            with col2:
                # Property selection for fuel mode
                viz_property = st.selectbox(
                    "Select Property to View",
                    [f"Property{i+1}" for i in range(10)],
                    key="viz_property"
                )
                bar_values = [p[viz_property] for p in props]
                blended_value = 123 #Modify

                fig2 = px.bar(
                    x=components,
                    y=bar_values,
                    title=f"{viz_property} Values",
                    color=bar_values,
                    color_continuous_scale='YlOrBr'
                )
                fig2.update_layout(
                    yaxis_title=viz_property,
                    xaxis_title="Component",
                    margin=dict(t=50, b=10),
                    coloraxis_showscale=False
                )
                
                fig2.add_hline(
                    y=blended_value,
                    line_dash="dot",
                    line_color="#ff6600",
                    annotation_text="Blended Value",
                    annotation_position="top right"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Display the calculated value prominently
                st.markdown(f"""
                    <div style="
                        background-color: #FAF3E6;
                        border-left: 4px solid #8B4513;
                        border-radius: 4px;
                        padding: 12px;
                        margin: 12px 0;
                    ">
                        <p style="margin: 0; color: #654321; 
                        font-size: 2.2rem;
                        font-weight: 800;
                        color: #000;
                        text-align:center;">
                            Calculated <strong>{viz_property}</strong> = 
                            <strong style="color: #000">{blended_value:.4f}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Batch mode visualization placeholder
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            st.subheader("Batch Processing Results")
            st.dataframe(data_input, use_container_width=True)
            # st.info("Batch processing complete. Add custom visualizations here.")


with tabs[2]:
    st.subheader("📤 Nothing FOr NOw")
    # uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    # if uploaded_file:
    #     df = pd.read_csv(uploaded_file)
    #     st.success("File uploaded successfully")
    #     st.dataframe(df.head())

    #     if st.button("⚙️ Run Batch Prediction"):
    #         result_df = df.copy()
    #         # result_df["Predicted_Property"] = df.apply(
    #         #     lambda row: run_dummy_prediction(row.values[:5], row.values[5:10]), axis=1
    #         # )
    #         st.success("Batch prediction completed")
    #         st.dataframe(result_df.head())
    #         csv = result_df.to_csv(index=False).encode("utf-8")
    #         st.download_button("Download Results", csv, "prediction_results.csv", "text/csv")



with tabs[3]:
    st.subheader("⚙️ Optimization Engine")
    
    # Pareto frontier demo
    st.markdown("#### Cost vs Performance Trade-off")
    np.random.seed(42)
    optimization_data = pd.DataFrame({
        'Cost ($/ton)': np.random.uniform(100, 300, 50),
        'Performance Score': np.random.uniform(70, 95, 50)
    })
    
    fig3 = px.scatter(
        optimization_data,
        x='Cost ($/ton)',
        y='Performance Score',
        title="Potential Blend Formulations",
        color='Performance Score',
        color_continuous_scale='YlOrBr'
    )
    
    # Add dummy pareto frontier
    x_pareto = np.linspace(100, 300, 10)
    y_pareto = 95 - 0.1*(x_pareto-100)
    fig3.add_trace(px.line(
        x=x_pareto,
        y=y_pareto,
        color_discrete_sequence= ['#8B4513', '#CFB53B', '#654321']
    ).data[0])
    
    fig3.update_layout(
        showlegend=False,
        annotations=[
            dict(
                x=200,
                y=88,
                text="Pareto Frontier",
                showarrow=True,
                arrowhead=1,
                ax=-50,
                ay=-30
            )
        ]
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Blend optimization history
    st.markdown("#### Optimization Progress")
    iterations = np.arange(20)
    performance = np.concatenate([np.linspace(70, 85, 10), np.linspace(85, 89, 10)])
    
    fig4 = px.line(
        x=iterations,
        y=performance,
        title="Best Performance by Iteration",
        markers=True
    )
    fig4.update_traces(
        line_color='#1d3b58',
        marker_color='#2c5282',
        line_width=2.5
    )
    fig4.update_layout(
        yaxis_title="Performance Score",
        xaxis_title="Iteration"
    )
    st.plotly_chart(fig4, use_container_width=True)



with tabs[4]:
    st.subheader("📚 Fuel Registry")  # Changed to book emoji for registry
    
    # Button to add new fuel
    st.markdown("#### ➕ Add a New Fuel Type")
    with st.expander("Click to Add New Fuel", expanded=False):
        with st.form("new_fuel_form", clear_on_submit=False):
            fuel_name = st.text_input("Fuel Name", placeholder="e.g. Bioethanol")
            
            cols = st.columns(5)
            properties = {}
            for i in range(10):
                with cols[i % 5]:
                    prop_val = st.number_input(
                        f"Property {i+1}", 
                        min_value=0.0, 
                        step=0.1, 
                        key=f"prop_{i}",
                        format="%.2f"
                    )
                    properties[f"Property{i+1}"] = round(prop_val, 2)

            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("💾 Save Fuel", use_container_width=True)
            with col2:
                cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)
            
            if submitted:
                if not fuel_name.strip():
                    st.warning("Fuel name cannot be empty.")
                elif fuel_name in st.session_state.FUEL_PROPERTIES:
                    st.error(f"{fuel_name} already exists in registry.")
                else:
                    # Update both session state and CSV
                    st.session_state.FUEL_PROPERTIES[fuel_name] = properties
                    save_fuel_data()
                    st.success(f"{fuel_name} successfully added!")
                    st.rerun()  # Refresh to show new fuel
            
            if cancelled:
                st.rerun()

    with st.expander("Batch Add New Fuel", expanded=False):
        uploaded_file = st.file_uploader(
            "📤 Upload Fuel Batch (CSV)",
            type=['csv'],
            accept_multiple_files=False,
            key="fuel_uploader",
            help="Upload a CSV file with the same format as the exported registry"
        )
        if uploaded_file is not None:
            try:
                new_fuels = pd.read_csv(uploaded_file, index_col=0).to_dict('index')
                
                # Check for duplicates
                duplicates = [name for name in new_fuels if name in st.session_state.FUEL_PROPERTIES]
                
                if duplicates:
                    st.warning(f"These fuels already exist and won't be updated: {', '.join(duplicates)}")
                    # Only add new fuels
                    new_fuels = {name: props for name, props in new_fuels.items() 
                                if name not in st.session_state.FUEL_PROPERTIES}
                
                if new_fuels:
                    st.session_state.FUEL_PROPERTIES.update(new_fuels)
                    save_fuel_data()
                    st.success(f"Added {len(new_fuels)} new fuel(s) to registry!")
                    st.rerun()
                else:
                    st.info("No new fuels to add from the uploaded file.")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.error("Please ensure the file matches the expected format")

    # Display current fuel properties
    st.markdown("#### 🔍 Current Fuel Properties")
    st.dataframe(
        pd.DataFrame(st.session_state.FUEL_PROPERTIES).T.style
        .background_gradient(cmap="YlOrBr", axis=None)
        .format(precision=2),
        use_container_width=True,
        height=(len(st.session_state.FUEL_PROPERTIES) + 1) * 35 + 3,
        hide_index=False
    )
    
    # File operations section

    
    st.download_button(
        label="📥 Download Registry (CSV)",
        data=pd.DataFrame(st.session_state.FUEL_PROPERTIES).T.to_csv().encode('utf-8'),
        file_name='fuel_properties.csv',
        mime='text/csv',
        # use_container_width=True
    )
    


        







with tabs[5]:
    st.subheader("🧠 Model Insights")
    
    # Feature importance
    st.markdown("#### Property Importance")
    features = ['Property 1', 'Property 2', 'Property 3', 'Property 4', 'Property 5']
    importance = np.array([0.35, 0.25, 0.2, 0.15, 0.05])
    
    fig5 = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance for Blend Prediction",
        color=importance,
        color_continuous_scale='YlOrBr'
    )
    fig5.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Property",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    # SHAP values demo
    st.markdown("#### Property Impact Direction")
    fig6 = px.scatter(
        x=np.random.randn(100),
        y=np.random.randn(100),
        color=np.random.choice(features, 100),
        title="SHAP Values (Simulated)",
        labels={'x': 'Impact on Prediction', 'y': 'Property Value'}
    )
    fig6.update_traces(
        marker=dict(size=10, opacity=0.7),
        selector=dict(mode='markers')
    )
    fig6.add_vline(x=0, line_width=1, line_dash="dash")
    st.plotly_chart(fig6, use_container_width=True)



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