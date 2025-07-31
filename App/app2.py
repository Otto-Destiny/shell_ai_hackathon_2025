import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot  as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
# from blend_logic import run_dummy_prediction

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
    
    /* Better select widget alignment */
    .stSelectbox > div {
        margin-bottom: -15px;
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
# with tabs[0]:
#     st.subheader("Performance Metrics")
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.markdown("""
#             <div class="metric-card">
#                 <h3 style='color: #1d3b58; margin-top: 0;'>Model Accuracy</h3>
#                 <p style='font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; color: #2c5282;'>94.7%</p>
#                 <p style='color: #6c757d; margin-bottom: 0;'>R² Score</p>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#             <div class="metric-card">
#                 <h3 style='color: #1d3b58; margin-top: 0;'>Predictions Made</h3>
#                 <p style='font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; color: #2c5282;'>12,847</p>
#                 <p style='color: #6c757d; margin-bottom: 0;'>Today</p>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         st.markdown("""
#             <div class="metric-card">
#                 <h3 style='color: #1d3b58; margin-top: 0;'>Optimizations</h3>
#                 <p style='font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; color: #2c5282;'>156</p>
#                 <p style='color: #6c757d; margin-bottom: 0;'>This Week</p>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col4:
#         st.markdown("""
#             <div class="metric-card">
#                 <h3 style='color: #1d3b58; margin-top: 0;'>Cost Savings</h3>
#                 <p style='font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; color: #2c5282;'>$2.4M</p>
#                 <p style='color: #6c757d; margin-bottom: 0;'>Estimated Annual</p>
#             </div>
#         """, unsafe_allow_html=True)
    

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


    # st.dataframe(
    #     df.style
    #     .set_table_styles([{
    #         'selector': '',
    #         'props': [
    #             ('width', '50%'),
    #             ('margin', '0 auto'),
    #             ('border-radius', '8px'),
    #             ('overflow', 'hidden')
    #         ]
    #     }])
    #     # Row alternation (zebra stripes)
    #     .apply(lambda x: [
    #         'background-color: #F5EBD8' if i%2 else 
    #         'background-color: #FAF3E6' 
    #         for i in range(len(x))], axis=0)
    #     # Column alternation (subtle)
    #     .apply(lambda x: [
    #         'border-left: 1px solid #E3D5B8' if i%2 else ''
    #         for i in range(len(x))], axis=1)
    #     # Header styling
    #     .set_properties(**{
    #         'color': '#654321',
    #         'border': '1px solid #E3D5B8'
    #     })
    #     .set_table_styles([{
    #         'selector': 'th',
    #         'props': [
    #             ('background-color', '#8B4513'),
    #             ('color', 'white'),
    #             ('font-weight', '600'),
    #             ('padding', '10px 15px')
    #         ]
    #     }])
    #     .format({"Value": "{:.3f}"}),
    #     use_container_width=True  # Important for custom width
    # )


#     df.style
# .background_gradient(cmap="YlOrBr", subset=["Value"])  /* Gold-orange gradient */
# .format({"Value": "{:.3f}"})

# ---------------------- Blend Designer Tab ----------------------
# with tabs[1]:
#     st.subheader("🎛️ Blend Designer")

#     # Property selection
#     selected_property_idx = st.selectbox(
#         "Select Property to Predict in Blend", 
#         [f"Property {i+1}" for i in range(10)],
#         key="property_select"
#     )
    
#     st.markdown("#### Component Weights and Selected Property Values")
    
#     weights, props = [], []
    
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("##### ⚖️ Component Weights")
#         for i in range(5):
#             weight = st.number_input(
#                 f"Weight for Component {i+1}", 
#                 min_value=0.0, 
#                 max_value=1.0, 
#                 value=0.2, 
#                 step=0.01, 
#                 key=f"w_{i}"
#             )
#             weights.append(weight)

#     with col2:
#         st.markdown(f"##### 🧪 {selected_property_idx} Values")
#         for i in range(5):
#             prop = st.number_input(
#                 f"Component {i+1} - {selected_property_idx}", 
#                 min_value=-6.0, 
#                 max_value=6.0, 
#                 value=0.5, 
#                 step=0.01, 
#                 key=f"p_{i}"
#             )
#             props.append(prop)

#     st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
#     if st.button("⚙️ Predict Blended Property", key="predict_btn"):
#         total_weight = sum(weights)
#         if abs(total_weight - 1.0) > 0.01:
#             st.warning("⚠️ The total of weights must be **1.0**.")
#         else:
#             blended_value = sum(w * p for w, p in zip(weights, props))
#             st.success(
#                 f"✅ Blended value for **{selected_property_idx}**: **{blended_value:.4f}**"
#             )


with tabs[1]:
    st.subheader("🎛️ Blend Designer")

    # Property selection
    selected_property_idx = st.selectbox(
        "Select Property to Predict in Blend", 
        [f"Property {i+1}" for i in range(10)],
        key="property_select"
    )
    
    st.markdown("#### Component Weights and Selected Property Values")
    
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
        st.markdown(f"##### � {selected_property_idx} Values")
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

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    if st.button("⚙️ Predict Blended Property", key="predict_btn"):
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 0.01:
            st.warning("⚠️ The total of weights must be **1.0**.")
        else:
            blended_value = sum(w * p for w, p in zip(weights, props))
            # st.success(f"✅ Blended value for **{selected_property_idx}**: **{blended_value:.4f}**")
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
                            Calculated <strong>{selected_property_idx}</strong> = 
                            <strong style="color: #000">{blended_value:.4f}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visualization Section
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
                    color_discrete_sequence= ['#8B4513', '#CFB53B', '#654321'], #.colors.sequential.Blues_r,
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
                fig2 = px.bar(
                    x=components,
                    y=props,
                    title=f"{selected_property_idx} Values",
                    color=props,
                    color_continuous_scale='YlOrBr'
                )
                fig2.update_layout(
                    yaxis_title=selected_property_idx,
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
            


with tabs[2]:
    st.subheader("📈 Property Analyzer")
    
    # Demo correlation matrix
    st.markdown("#### Property Correlations")
    np.random.seed(42)
    demo_data = pd.DataFrame({
        'Property 1': np.random.normal(50, 10, 100),
        'Property 2': np.random.normal(8, 2, 100),
        'Property 3': np.random.normal(0.85, 0.05, 100),
        'Property 4': np.random.normal(45, 3, 100)
    })
    
    fig1 = px.imshow(
        demo_data.corr(),
        color_continuous_scale='YlOrBr',
        title="Property Correlation Matrix",
        zmin=-1,
        zmax=1
    )
    fig1.update_layout(width=600, height=500)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Property trend visualization
    st.markdown("#### Property Relationships")
    x_prop = st.selectbox("X-axis Property", demo_data.columns, index=0)
    y_prop = st.selectbox("Y-axis Property", demo_data.columns, index=1)
    
    fig2 = px.scatter(
        demo_data,
        x=x_prop,
        y=y_prop,
        trendline="lowess",
        title=f"{x_prop} vs {y_prop} Relationship",
        color_discrete_sequence= ['#8B4513', '#CFB53B', '#654321']
    )
    fig2.update_traces(
        marker=dict(size=8, opacity=0.7),
        line=dict(color='#ff6600', width=3)
    )
    st.plotly_chart(fig2, use_container_width=True)



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
    st.subheader("📤 Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully")
        st.dataframe(df.head())

        if st.button("⚙️ Run Batch Prediction"):
            result_df = df.copy()
            # result_df["Predicted_Property"] = df.apply(
            #     lambda row: run_dummy_prediction(row.values[:5], row.values[5:10]), axis=1
            # )
            st.success("Batch prediction completed")
            st.dataframe(result_df.head())
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "prediction_results.csv", "text/csv")

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