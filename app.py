import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Fuel Blend Optimizer Pro",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .optimization-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>Eagle Blend Optimizer Pro</h1>
    <h3>AI-Powered Fuel Blend Property Prediction & Optimization</h3>
    <p>Shell AI Hackathon Phase 2 Solution | Eagle-Team</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("🎛️ Control Panel")
page = st.sidebar.selectbox("Navigate", [
    "🏠 Dashboard", 
    "🧪 Blend Designer", 
    "📊 Property Analyzer",
    "🎯 Optimization Engine",
    "📈 Batch Processing",
    "🔬 Model Insights"
])

# Generate dummy data for demo
@st.cache_data
def generate_dummy_data():
    np.random.seed(42)
    
    # Component fractions (5 components)
    components = ['Component_A', 'Component_B', 'Component_C', 'Component_D', 'Component_E']
    
    # 10 properties for each component (anonymized)
    properties = [f'Property_{i+1}' for i in range(10)]
    
    # Target blend properties (10 properties)
    target_properties = [f'BlendProp_{i+1}' for i in range(10)]
    
    return components, properties, target_properties

components, properties, target_properties = generate_dummy_data()

if page == "🏠 Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>🎯 Model Accuracy</h3>
            <h2>94.7%</h2>
            <p>R² Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3>⚡ Predictions Made</h3>
            <h2>12,847</h2>
            <p>Today</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3>🔧 Optimizations</h3>
            <h2>156</h2>
            <p>This Week</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <h3>💰 Cost Savings</h3>
            <h2>$2.4M</h2>
            <p>Estimated Annual</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Real-time monitoring
    st.subheader("📊 Real-Time Blend Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate dummy time series data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
        values = 92 + np.random.normal(0, 2, len(dates)).cumsum() * 0.1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='Blend Quality Score',
                                line=dict(color='#FF6B35', width=3)))
        fig.update_layout(title="Quality Score Trend (Last 30 Days)", 
                         xaxis_title="Time", yaxis_title="Quality Score (%)",
                         height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Property distribution
        np.random.seed(42)
        prop_values = np.random.normal(0.5, 0.15, 10)
        prop_names = [f'Property {i+1}' for i in range(10)]
        
        fig = go.Figure(data=[
            go.Bar(x=prop_names, y=prop_values, 
                  marker_color=['#FF6B35', '#F7931E', '#667eea', '#764ba2', '#f093fb',
                               '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'])
        ])
        fig.update_layout(title="Current Blend Properties Distribution",
                         xaxis_title="Properties", yaxis_title="Normalized Values",
                         height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

elif page == "🧪 Blend Designer":
    st.markdown("""
    <div class="prediction-card">
        <h2>🧪 Interactive Blend Designer</h2>
        <p>Design your optimal fuel blend with real-time AI predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎛️ Component Mixer")
        
        # Component fraction sliders
        fractions = {}
        remaining = 1.0
        
        for i, comp in enumerate(components[:-1]):
            max_val = min(remaining, 1.0)
            fractions[comp] = st.slider(f"{comp} Fraction", 0.0, max_val, max_val/len(components), 0.01)
            remaining -= fractions[comp]
        
        fractions[components[-1]] = max(0, remaining)
        st.write(f"**{components[-1]} Fraction: {fractions[components[-1]]:.3f}** (Auto-calculated)")
        
        # Validation
        total = sum(fractions.values())
        if abs(total - 1.0) > 0.001:
            st.error(f"⚠️ Fractions must sum to 1.0 (Current: {total:.3f})")
        else:
            st.success("✅ Valid blend composition")
        
        # Prediction button
        if st.button("🔮 Predict Properties", type="primary"):
            with st.spinner("🤖 AI is analyzing your blend..."):
                time.sleep(2)  # Simulate processing
                st.success("✨ Prediction complete!")
    
    with col2:
        st.subheader("📊 Predicted Blend Properties")
        
        # Generate dummy predictions
        np.random.seed(int(sum(fractions.values()) * 1000))
        predictions = np.random.normal(0.5, 0.2, 10)
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=predictions,
            theta=target_properties,
            fill='toself',
            name='Predicted Properties',
            line_color='#FF6B35'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            height=500,
            title="Blend Property Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Property table
        pred_df = pd.DataFrame({
            'Property': target_properties,
            'Predicted Value': predictions,
            'Confidence': np.random.uniform(0.85, 0.98, 10),
            'Target Range': ['0.4-0.6'] * 10
        })
        
        st.dataframe(pred_df.style.format({
            'Predicted Value': '{:.4f}',
            'Confidence': '{:.1%}'
        }), use_container_width=True)

elif page == "📊 Property Analyzer":
    st.markdown("""
    <div class="optimization-card">
        <h2>📊 Advanced Property Analyzer</h2>
        <p>Deep dive into component-property relationships and correlations</p>
    </div>
    """, unsafe_allow_html=True)
    
    analysis_type = st.selectbox("Analysis Type", [
        "Component Impact Analysis",
        "Property Correlation Matrix",
        "Sensitivity Analysis",
        "Historical Trends"
    ])
    
    if analysis_type == "Component Impact Analysis":
        col1, col2 = st.columns(2)
        
        with col1:
            selected_property = st.selectbox("Select Target Property", target_properties)
            
            # Generate impact data
            impact_data = np.random.uniform(0.1, 0.9, len(components))
            
            fig = go.Figure(data=[
                go.Bar(x=components, y=impact_data,
                      marker_color=['#FF6B35', '#F7931E', '#667eea', '#764ba2', '#f093fb'])
            ])
            fig.update_layout(title=f"Component Impact on {selected_property}",
                             xaxis_title="Components", yaxis_title="Impact Score",
                             height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance heatmap
            np.random.seed(42)
            importance_matrix = np.random.rand(len(components), len(target_properties))
            
            fig = px.imshow(importance_matrix,
                           x=target_properties,
                           y=components,
                           color_continuous_scale='Viridis',
                           title="Component-Property Importance Matrix")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Property Correlation Matrix":
        # Generate correlation matrix
        np.random.seed(42)
        corr_matrix = np.random.rand(len(target_properties), len(target_properties))
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1)  # Diagonal = 1
        
        fig = px.imshow(corr_matrix,
                       x=target_properties,
                       y=target_properties,
                       color_continuous_scale='RdBu',
                       color_continuous_midpoint=0,
                       title="Blend Property Correlation Matrix")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Optimization Engine":
    st.markdown("""
    <div class="optimization-card">
        <h2>🎯 Multi-Objective Optimization Engine</h2>
        <p>Find the optimal blend composition for your specific requirements</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Optimization Goals")
        
        # Optimization objectives
        objectives = {}
        for prop in target_properties[:5]:  # Show first 5 for demo
            col_a, col_b = st.columns(2)
            with col_a:
                objectives[f"{prop}_target"] = st.number_input(f"{prop} Target", 0.0, 1.0, 0.5, key=f"target_{prop}")
            with col_b:
                objectives[f"{prop}_weight"] = st.slider(f"Weight", 0.1, 2.0, 1.0, key=f"weight_{prop}")
        
        # Constraints
        st.subheader("⚖️ Constraints")
        cost_constraint = st.checkbox("Minimize Cost", True)
        environmental_constraint = st.checkbox("Environmental Compliance", True)
        availability_constraint = st.checkbox("Component Availability", True)
        
        # Optimization algorithm
        algorithm = st.selectbox("Algorithm", ["Genetic Algorithm", "Particle Swarm", "Bayesian Optimization"])
        
        if st.button("🚀 Start Optimization", type="primary"):
            with st.spinner("🧠 Optimizing blend composition..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                st.success("✨ Optimization Complete!")
    
    with col2:
        st.subheader("📈 Optimization Results")
        
        # Pareto front visualization
        np.random.seed(42)
        pareto_x = np.random.rand(50)
        pareto_y = 1 - pareto_x + np.random.normal(0, 0.1, 50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pareto_x, y=pareto_y, mode='markers',
                                marker=dict(size=8, color='#FF6B35'),
                                name='Pareto Solutions'))
        
        # Highlight optimal solution
        optimal_idx = np.argmax(pareto_x + pareto_y)
        fig.add_trace(go.Scatter(x=[pareto_x[optimal_idx]], y=[pareto_y[optimal_idx]],
                                mode='markers', marker=dict(size=15, color='red', symbol='star'),
                                name='Recommended Solution'))
        
        fig.update_layout(title="Multi-Objective Optimization Results",
                         xaxis_title="Objective 1 (Quality Score)",
                         yaxis_title="Objective 2 (Cost Efficiency)",
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimal composition
        st.subheader("🏆 Recommended Blend Composition")
        optimal_fractions = np.random.dirichlet(np.ones(5))
        optimal_df = pd.DataFrame({
            'Component': components,
            'Optimal Fraction': optimal_fractions,
            'Current Market Price': np.random.uniform(50, 200, 5),
            'Availability': np.random.choice(['High', 'Medium', 'Low'], 5)
        })
        
        st.dataframe(optimal_df.style.format({
            'Optimal Fraction': '{:.4f}',
            'Current Market Price': '${:.2f}/barrel'
        }), use_container_width=True)

elif page == "📈 Batch Processing":
    st.markdown("""
    <div class="prediction-card">
        <h2>📈 Batch Processing & Analysis</h2>
        <p>Process multiple blend scenarios and compare results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload simulation
    st.subheader("📁 Upload Blend Scenarios")
    uploaded_file = st.file_uploader("Choose CSV file with blend compositions", type=['csv'])
    
    if st.button("Generate Sample Data"):
        # Create sample batch data
        np.random.seed(42)
        n_scenarios = 100
        batch_data = pd.DataFrame()
        
        for comp in components:
            batch_data[comp] = np.random.dirichlet(np.ones(5), n_scenarios)[:, components.index(comp)]
        
        for prop in target_properties:
            batch_data[f"Predicted_{prop}"] = np.random.normal(0.5, 0.2, n_scenarios)
        
        batch_data['Scenario_ID'] = range(1, n_scenarios + 1)
        batch_data['Quality_Score'] = np.random.normal(85, 10, n_scenarios)
        batch_data['Cost_Estimate'] = np.random.normal(150, 30, n_scenarios)
        
        st.success(f"✅ Generated {n_scenarios} blend scenarios!")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality vs Cost scatter
            fig = px.scatter(batch_data, x='Cost_Estimate', y='Quality_Score',
                           color='Quality_Score', size='Quality_Score',
                           title="Quality vs Cost Analysis",
                           labels={'Cost_Estimate': 'Estimated Cost ($/barrel)',
                                  'Quality_Score': 'Quality Score'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Component distribution
            component_cols = [col for col in batch_data.columns if col in components]
            fig = go.Figure()
            for comp in component_cols:
                fig.add_trace(go.Histogram(x=batch_data[comp], name=comp, opacity=0.7))
            fig.update_layout(title="Component Fraction Distributions",
                            xaxis_title="Fraction", yaxis_title="Frequency",
                            barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)
        
        # Top performers table
        st.subheader("🏆 Top 10 Performing Blends")
        top_blends = batch_data.nlargest(10, 'Quality_Score')[['Scenario_ID'] + components + ['Quality_Score', 'Cost_Estimate']]
        st.dataframe(top_blends.style.format({
            col: '{:.4f}' for col in components
        } | {'Quality_Score': '{:.2f}', 'Cost_Estimate': '${:.2f}'}), use_container_width=True)

elif page == "🔬 Model Insights":
    st.markdown("""
    <div class="optimization-card">
        <h2>🔬 Model Insights & Explainability</h2>
        <p>Understand how your AI model makes predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🎯 SHAP Analysis", "🧠 Model Performance"])
    
    with tab1:
        # Feature importance visualization
        np.random.seed(42)
        feature_names = []
        for comp in components:
            for prop in properties:
                feature_names.append(f"{comp}_{prop}")
        
        importance_scores = np.random.exponential(1, len(feature_names))
        importance_scores = importance_scores / importance_scores.sum()
        
        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        top_features = [feature_names[i] for i in sorted_indices[:20]]
        top_scores = [importance_scores[i] for i in sorted_indices[:20]]
        
        fig = go.Figure(data=[
            go.Bar(x=top_scores, y=top_features, orientation='h',
                  marker_color='#FF6B35')
        ])
        fig.update_layout(title="Top 20 Feature Importance Scores",
                         xaxis_title="Importance Score",
                         yaxis_title="Features",
                         height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.info("🔍 SHAP (SHapley Additive exPlanations) values show how each feature contributes to individual predictions")
        
        # Simulated SHAP waterfall chart
        features = ['Component_A_Prop1', 'Component_B_Prop3', 'Component_C_Prop7', 'Component_A_Prop2', 'Component_D_Prop5']
        shap_values = [0.15, -0.08, 0.12, -0.05, 0.09]
        base_value = 0.45
        
        fig = go.Figure(go.Waterfall(
            name="SHAP Values",
            orientation="v",
            measure=["relative"] * len(features) + ["total"],
            x=features + ["Final Prediction"],
            textposition="outside",
            text=[f"{val:+.3f}" for val in shap_values] + [f"{base_value + sum(shap_values):.3f}"],
            y=shap_values + [base_value + sum(shap_values)],
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig.update_layout(title="SHAP Waterfall Plot - Sample Prediction",
                         showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Model performance metrics
            st.subheader("📈 Model Performance Metrics")
            
            metrics_data = {
                'Metric': ['R² Score', 'RMSE', 'MAE', 'MAPE'],
                'Value': [0.947, 0.023, 0.018, '2.3%'],
                'Benchmark': [0.920, 0.035, 0.025, '3.1%'],
                'Status': ['🟢 Excellent', '🟢 Good', '🟢 Good', '🟢 Good']
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Residual plot
            np.random.seed(42)
            y_true = np.random.normal(0.5, 0.2, 100)
            y_pred = y_true + np.random.normal(0, 0.05, 100)
            residuals = y_true - y_pred
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                                   marker=dict(color='#FF6B35', size=6),
                                   name='Residuals'))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.update_layout(title="Residual Plot",
                             xaxis_title="Predicted Values",
                             yaxis_title="Residuals",
                             height=400)
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🏆 Shell AI Hackathon Phase 2 | Built with ❤️ using Streamlit & AI</p>
    <p>Revolutionizing Energy with Artificial Intelligence | Eagle Team</p>
</div>
""", unsafe_allow_html=True)
