import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import streamlit as st
import plotly.express as px
from io import StringIO
import time
import shap
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
st.set_page_config(
    page_title="PharmaRelease AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://example.com',
        'Report a bug': "https://example.com",
        'About': "# Advanced Drug Release Prediction System"
    }
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# Cache data loading and preprocessing
@st.cache_data
def load_data():
    df = pd.read_csv('taguchi1.csv').rename(columns={
        'Run ': 'Run',
        'perc of Drug Release': 'Drug_Release'
    })
    return df

@st.cache_resource
def train_base_models(X, y):
    """Train and cache all base models"""
    models = {}
    
    # Linear Models
    models['Linear'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]).fit(X, y)
    
    models['Ridge'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=0.1))
    ]).fit(X, y)
    
    models['Lasso'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=0.1))
    ]).fit(X, y)
    
    # Polynomial Regression
    models['Poly'] = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', LinearRegression())
    ]).fit(X, y)
    
    # Tree-based Models
    models['Decision Tree'] = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X, y)
    models['Random Forest'] = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1).fit(X, y)
    models['GBM'] = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y)
    
    # Boosted Trees
    models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1).fit(X, y)
    models['LightGBM'] = LGBMRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1).fit(X, y)
    models['CatBoost'] = CatBoostRegressor(iterations=100, depth=3, random_state=42, verbose=0).fit(X, y)
    
    # Support Vector Machine
    models['SVR'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ]).fit(X, y)
    
    # Neural Network
    models['MLP'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPRegressor(
            hidden_layer_sizes=(100,50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        ))
    ]).fit(X, y)
    
    return models

def train_ensemble_models(_base_models, X, y):
    """Train ensemble models without caching the base models input"""
    ensembles = {}
    
    # Stacking Ensemble
    estimators = [
        ('rf', _base_models['Random Forest']),
        ('xgb', _base_models['XGBoost']),
        ('svr', _base_models['SVR'])
    ]
    ensembles['Stacking'] = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        n_jobs=-1
    ).fit(X, y)
    
    # Bagging Ensemble
    ensembles['Bagging'] = BaggingRegressor(
        estimator=_base_models['Decision Tree'],
        n_estimators=10,
        random_state=42,
        n_jobs=-1
    ).fit(X, y)
    
    return ensembles

# Load data
df = load_data()
X = df.drop(columns=['Run', 'Drug_Release'])
y = df['Drug_Release']

# Train models (cached)
base_models = train_base_models(X, y)
ensemble_models = train_ensemble_models(base_models, X, y)
all_models = {**base_models, **ensemble_models}

# UI Components
def model_selection_ui():
    st.sidebar.header("🧪 Model Configuration")
    
    # Add logo
    st.sidebar.image("https://via.placeholder.com/300x100?text=PharmaRelease+AI", use_container_width=True)
    
    model_categories = {
        "Linear Models": ["Linear", "Ridge", "Lasso", "Poly"],
        "Tree-based Models": ["Decision Tree", "Random Forest", "GBM"],
        "Boosted Trees": ["XGBoost", "LightGBM", "CatBoost"],
        "Other Models": ["SVR", "MLP"],
        "Ensemble Models": ["Stacking", "Bagging"]
    }
    
    selected_category = st.sidebar.selectbox(
        "Model Category",
        list(model_categories.keys()),
        key="model_category"
    )
    
    model_option = st.sidebar.selectbox(
        "Select Model", 
        model_categories[selected_category],
        key="model_option"
    )
    
    # Model-specific parameters
    st.sidebar.markdown("### 🛠️ Model Parameters")
    if model_option in ["Random Forest", "GBM", "XGBoost", "LightGBM", "CatBoost"]:
        n_estimators = st.sidebar.slider(
            "Number of estimators",
            50, 500, 
            value=100,
            step=50,
            key=f"{model_option}_n_estimators"
        )
        max_depth = st.sidebar.slider(
            "Max depth", 
            3, 10, 5,
            key=f"{model_option}_max_depth"
        )
        
        if model_option == "Random Forest":
            all_models[model_option] = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            ).fit(X, y)
        elif model_option == "XGBoost":
            all_models[model_option] = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            ).fit(X, y)
    
    elif model_option == "SVR":
        C = st.sidebar.slider(
            "Regularization (C)", 
            0.1, 10.0, 1.0, 0.1,
            key="svr_c"
        )
        epsilon = st.sidebar.slider(
            "Epsilon", 
            0.01, 1.0, 0.1, 0.01,
            key="svr_epsilon"
        )
        all_models[model_option] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=C, epsilon=epsilon))
        ]).fit(X, y)
    
    elif model_option == "MLP":
        hidden_layers = st.sidebar.slider(
            "Hidden layer size", 
            10, 200, 100,
            key="mlp_size"
        )
        all_models[model_option] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(hidden_layers, hidden_layers//2),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            ))
        ]).fit(X, y)
    
    return model_option

def input_parameters_ui():
    st.header("🧪 Input Parameters")
    cols = st.columns(3)
    inputs = {}
    
    with cols[0]:
        st.markdown("### ⏱️ Time & Concentration")
        inputs['Time(min)'] = st.slider(
            "Time (min)", 
            float(X['Time(min)'].min()), 
            float(X['Time(min)'].max(),
            float(X['Time(min)'].median()),
            step=1.0,
            key="time_input"
        )
        inputs['Drug_con(Mg)'] = st.slider(
            "Drug Concentration (Mg)", 
            float(X['Drug_con(Mg)'].min()), 
            float(X['Drug_con(Mg)'].max()),
            float(X['Drug_con(Mg)'].median()),
            step=0.1,
            key="conc_input"
        )
    
    with cols[1]:
        st.markdown("### 🔄 Mixing Parameters")
        inputs['Rpm'] = st.slider(
            "RPM", 
            int(X['Rpm'].min()), 
            int(X['Rpm'].max()),
            int(X['Rpm'].median()),
            step=10,
            key="rpm_input"
        )
        inputs['pH'] = st.slider(
            "pH", 
            float(X['pH'].min()), 
            float(X['pH'].max()),
            float(X['pH'].median()),
            step=0.1,
            key="ph_input"
        )
    
    with cols[2]:
        st.markdown("### 🌡️ Environmental")
        inputs['Temperature'] = st.slider(
            "Temperature (°C)", 
            int(X['Temperature'].min()), 
            int(X['Temperature'].max()),
            int(X['Temperature'].median()),
            step=1,
            key="temp_input"
        )
        st.markdown("---")
        st.markdown("### ⚡ Quick Presets")
        preset_cols = st.columns(2)
        with preset_cols[0]:
            if st.button("🔘 Default Settings", help="Reset to median values"):
                st.session_state.time_input = float(X['Time(min)'].median())
                st.session_state.conc_input = float(X['Drug_con(Mg)'].median())
                st.session_state.rpm_input = int(X['Rpm'].median())
                st.session_state.ph_input = float(X['pH'].median())
                st.session_state.temp_input = int(X['Temperature'].median())
        with preset_cols[1]:
            if st.button("🚀 Optimal (Max Release)", help="Set parameters for maximum release"):
                st.session_state.time_input = float(X['Time(min)'].max())
                st.session_state.conc_input = float(X['Drug_con(Mg)'].median())
                st.session_state.rpm_input = int(X['Rpm'].max())
                st.session_state.ph_input = float(X['pH'].median())
                st.session_state.temp_input = int(X['Temperature'].median())
    
    return pd.DataFrame([inputs])

def show_prediction_results(model_name, X_input, y_true=None):
    """Display prediction results with visual feedback"""
    
    # Make prediction
    with st.spinner("🔮 Making prediction..."):
        start_time = time.time()
        model = all_models[model_name]
        prediction = model.predict(X_input)[0]
        pred_time = time.time() - start_time
    
    # Display results in a nice card
    with st.container():
        st.markdown("### 📊 Prediction Results")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.metric(
                "Predicted Release", 
                f"{prediction:.2f}%",
                delta="Optimal" if prediction > 70 else "Suboptimal",
                delta_color="normal"
            )
            
            # Visual indicator
            release_gauge = st.progress(int(prediction))
            st.caption(f"Model: {model_name}")
        
        with col2:
            # Performance metrics
            if y_true is not None:
                y_pred = model.predict(X)
                mse = mean_squared_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                metric_cols = st.columns(3)
                metric_cols[0].metric("R² Score", f"{r2:.3f}", "Accuracy")
                metric_cols[1].metric("MSE", f"{mse:.3f}", "Error")
                metric_cols[2].metric("Time", f"{pred_time:.4f}s", "Speed")
    
    # SHAP explanation
    if model_name in ["Decision Tree", "Random Forest", "XGBoost", "LightGBM"]:
        try:
            st.markdown("---")
            st.markdown("### 🔍 Feature Impact Analysis")
            
            tab1, tab2 = st.tabs(["Summary", "Current Prediction"])
            
            with tab1:
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)
                
                fig_shap, ax = plt.subplots()
                shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                st.pyplot(fig_shap)
            
            with tab2:
                shap_input = pd.DataFrame(X_input, columns=X.columns)
                shap_values_single = explainer(shap_input)
                
                fig_force, ax = plt.subplots()
                shap.plots.waterfall(shap_values_single[0], max_display=10, show=False)
                st.pyplot(fig_force)
        except Exception as e:
            st.warning(f"⚠️ Could not generate SHAP explanation: {str(e)}")
    
    # Recommendation
    st.markdown("---")
    if prediction < 30:
        st.warning("""
        ## ⚠️ Low Release Predicted
        **Recommendations:**
        - Increase mixing time (Time)
        - Increase agitation speed (RPM)
        - Check temperature settings
        """)
    elif prediction > 80:
        st.success("""
        ## ✅ High Release Predicted
        **Considerations:**
        - May need to adjust for sustained release
        - Check if release is too fast for target profile
        """)
    
    return prediction

def optimization_ui():
    st.header("⚡ Parameter Optimization")
    
    with st.expander("🔍 Optimization Settings", expanded=True):
        opt_target = st.radio(
            "Optimization Target",
            ["Maximize Drug Release", "Minimize Drug Release", "Target Release"],
            horizontal=True,
            key="opt_target"
        )
        
        target_value = None
        if opt_target == "Target Release":
            target_value = st.slider(
                "Target Release Percentage",
                float(y.min()), float(y.max()), float(y.mean()), 1.0,
                key="target_value"
            )
        
        model_for_opt = st.selectbox(
            "Model for Optimization",
            ["XGBoost", "Random Forest", "Stacking"],
            index=0,
            key="opt_model"
        )
    
    if st.button("🚀 Run Optimization", type="primary", help="Find optimal parameters using Bayesian optimization"):
        with st.spinner("🧠 Finding optimal parameters..."):
            try:
                # Define search space
                search_spaces = {
                    'Time(min)': Real(X['Time(min)'].min(), X['Time(min)'].max()),
                    'Drug_con(Mg)': Real(X['Drug_con(Mg)'].min(), X['Drug_con(Mg)'].max()),
                    'Rpm': Integer(X['Rpm'].min(), X['Rpm'].max()),
                    'pH': Real(X['pH'].min(), X['pH'].max()),
                    'Temperature': Integer(X['Temperature'].min(), X['Temperature'].max())
                }
                
                # Initialize a fresh model
                if model_for_opt == "XGBoost":
                    base_model = XGBRegressor(random_state=42)
                elif model_for_opt == "Random Forest":
                    base_model = RandomForestRegressor(random_state=42)
                else:
                    base_model = StackingRegressor(
                        estimators=[
                            ('rf', RandomForestRegressor(random_state=42)),
                            ('xgb', XGBRegressor(random_state=42))
                        ],
                        final_estimator=LinearRegression()
                    )
                
                # Define scoring
                if "Maximize" in opt_target:
                    scoring = 'neg_mean_squared_error'
                elif "Minimize" in opt_target:
                    scoring = 'neg_mean_squared_error'
                else:
                    def custom_scorer(estimator, X, y):
                        preds = estimator.predict(X)
                        return -np.mean(np.abs(preds - target_value))
                    scoring = custom_scorer
                
                # Run Bayesian optimization
                opt = BayesSearchCV(
                    estimator=base_model,
                    search_spaces=search_spaces,
                    n_iter=30,
                    cv=3,
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=42
                )
                
                opt.fit(X, y)
                
                # Get best parameters
                best_params = opt.best_params_
                best_value = opt.best_estimator_.predict(pd.DataFrame([best_params]))[0]
                
                # Display results
                st.success("## 🎯 Optimal Parameters Found")
                
                # Create a nice display of optimal parameters
                st.markdown("### ⚙️ Recommended Settings")
                param_cols = st.columns(5)
                params_display = {
                    'Time(min)': f"⏱️ {best_params['Time(min)']:.1f} min",
                    'Drug_con(Mg)': f"🧪 {best_params['Drug_con(Mg)']:.2f} Mg",
                    'Rpm': f"🌀 {best_params['Rpm']} RPM",
                    'pH': f"🧪 pH {best_params['pH']:.1f}",
                    'Temperature': f"🌡️ {best_params['Temperature']}°C"
                }
                
                for i, (param, display) in enumerate(params_display.items()):
                    param_cols[i].metric(
                        param.split('(')[0],
                        display,
                        f"Range: {X[param].min():.1f}-{X[param].max():.1f}"
                    )
                
                # Show optimization result
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Predicted Release", 
                        f"{best_value:.2f}%",
                        delta=f"Target: {target_value:.2f}%" if target_value else None
                    )
                with col2:
                    st.metric(
                        "Model Used",
                        model_for_opt
                    )
                
                # Visualize parameter space
                st.markdown("### 📊 Parameter Space Exploration")
                fig = px.scatter_3d(
                    df,
                    x='Time(min)',
                    y='Rpm',
                    z='Drug_Release',
                    color='Drug_Release',
                    hover_name='Run',
                    title="Optimization Landscape"
                )
                fig.add_scatter3d(
                    x=[best_params['Time(min)']],
                    y=[best_params['Rpm']],
                    z=[best_value],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='Optimal'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")

def model_comparison_ui():
    st.header("📊 Model Comparison Dashboard")
    
    with st.expander("⚙️ Comparison Settings", expanded=True):
        selected_models = st.multiselect(
            "Select models to compare",
            list(all_models.keys()),
            default=["Linear", "Random Forest", "XGBoost", "Stacking"],
            key="model_compare_select"
        )
    
    if st.button("🔍 Compare Models", type="primary"):
        with st.spinner("📊 Analyzing model performance..."):
            comparison_results = []
            
            for model_name in selected_models:
                model = all_models[model_name]
                y_pred = model.predict(X)
                
                # Calculate metrics
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                
                comparison_results.append({
                    'Model': model_name,
                    'R²': r2,
                    'MSE': mse,
                    'CV R² Mean': np.mean(cv_scores),
                    'CV R² Std': np.std(cv_scores)
                })
            
            # Display results
            comparison_df = pd.DataFrame(comparison_results)
            
            st.markdown("### 📈 Performance Metrics")
            st.dataframe(
                comparison_df.sort_values('R²', ascending=False),
                hide_index=True,
                use_container_width=True
            )
            
            # Visualization
            st.markdown("### 📊 Model Performance Comparison")
            tab1, tab2 = st.tabs(["R² Scores", "Error Analysis"])
            
            with tab1:
                fig_r2 = px.bar(
                    comparison_df,
                    x='Model',
                    y='R²',
                    error_y='CV R² Std',
                    title='Model Accuracy (R² Score)',
                    color='Model',
                    text='R²'
                )
                fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                st.plotly_chart(fig_r2, use_container_width=True)
            
            with tab2:
                fig_mse = px.bar(
                    comparison_df,
                    x='Model',
                    y='MSE',
                    title='Mean Squared Error',
                    color='Model',
                    text='MSE'
                )
                fig_mse.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                st.plotly_chart(fig_mse, use_container_width=True)
            
            # Actual vs Predicted plots
            st.markdown("### 🔍 Prediction Analysis")
            for model_name in selected_models:
                with st.expander(f"📌 {model_name} Details"):
                    model = all_models[model_name]
                    y_pred = model.predict(X)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.scatter(
                            x=y, y=y_pred,
                            labels={'x': 'Actual', 'y': 'Predicted'},
                            title=f"{model_name} Predictions",
                            trendline="lowess"
                        )
                        fig.add_shape(
                            type="line", 
                            x0=y.min(), y0=y.min(), 
                            x1=y.max(), y1=y.max(),
                            line=dict(color="Red", width=2, dash="dash")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        residuals = y - y_pred
                        fig_resid = px.scatter(
                            x=y_pred, y=residuals,
                            labels={'x': 'Predicted', 'y': 'Residuals'},
                            title=f"{model_name} Residuals"
                        )
                        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_resid, use_container_width=True)

def main():
    # Sidebar
    model_option = model_selection_ui()
    
    # Main content
    st.title("💊 PharmaRelease AI")
    st.markdown("""
    **Advanced Drug Release Prediction System**  
    *Predict and optimize drug release profiles using machine learning*
    """)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Prediction", 
        "📊 Analysis", 
        "⚡ Optimization", 
        "📈 Model Comparison",
        "⚙️ System"
    ])
    
    with tab1:
        input_df = input_parameters_ui()
        if st.button("🔮 Predict Drug Release", type="primary", use_container_width=True):
            prediction = show_prediction_results(model_option, input_df, y)
            
            # Show historical similar runs
            st.markdown("---")
            st.markdown("### 📚 Similar Historical Runs")
            df['similarity'] = np.sqrt(
                ((df[['Time(min)', 'Drug_con(Mg)', 'Rpm', 'pH', 'Temperature']] - input_df.values) ** 2).sum(axis=1))
            similar_runs = df.nsmallest(3, 'similarity')
            st.dataframe(
                similar_runs.drop(columns=['similarity']),
                use_container_width=True
            )
    
    with tab2:
        st.header("🔍 Data Analysis")
        
        analysis_type = st.radio(
            "Analysis Type",
            ["Parameter Relationships", "Model Performance", "Data Distribution"],
            horizontal=True,
            key="analysis_type"
        )
        
        if analysis_type == "Parameter Relationships":
            st.markdown("### 📈 Parameter Relationships")
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", X.columns, key="x_axis")
            with col2:
                y_axis = st.selectbox("Y-axis", ['Drug_Release'] + list(X.columns), key="y_axis")
            
            fig = px.scatter(
                df, 
                x=x_axis, 
                y=y_axis, 
                color='Drug_Release',
                hover_data=['Run'],
                title=f"{x_axis} vs {y_axis}",
                trendline="lowess"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Model Performance":
            y_pred = all_models[model_option].predict(X)
            fig = px.scatter(
                x=y, y=y_pred,
                labels={'x': 'Actual', 'y': 'Predicted'},
                title="Actual vs Predicted Drug Release",
                trendline="lowess"
            )
            fig.add_shape(
                type="line", 
                x0=y.min(), y0=y.min(), 
                x1=y.max(), y1=y.max(),
                line=dict(color="Red", width=2, dash="dash")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.markdown("### 📊 Data Distribution")
            selected_param = st.selectbox(
                "Select Parameter", 
                X.columns,
                key="dist_param"
            )
            fig = px.histogram(
                df, 
                x=selected_param,
                nbins=20,
                title=f"Distribution of {selected_param}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        optimization_ui()
    
    with tab4:
        model_comparison_ui()
    
    with tab5:
        st.header("⚙️ System Settings")
        
        with st.expander("🔧 Data Management", expanded=True):
            st.markdown("### 🔄 Data Operations")
            if st.button("🔄 Reload Data", help="Clear cache and reload data"):
                st.cache_data.clear()
                st.rerun()
            
            st.markdown("### 💾 Export Results")
            export_type = st.selectbox(
                "Export Type",
                ["Current Predictions", "Model Details", "Optimization Results"],
                key="export_type"
            )
            
            if st.button("📤 Generate Export", type="primary"):
                if export_type == "Current Predictions":
                    y_pred = all_models[model_option].predict(X)
                    results = df.copy()
                    results['Predicted'] = y_pred
                    csv = results.to_csv(index=False)
                    
                    st.download_button(
                        "💾 Download Predictions (CSV)",
                        csv,
                        "drug_release_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                elif export_type == "Model Details":
                    buffer = StringIO()
                    buffer.write(f"Drug Release Prediction Model\n")
                    buffer.write(f"Trained on: {pd.Timestamp.now()}\n\n")
                    
                    for name, model in all_models.items():
                        buffer.write(f"=== {name.upper()} ===\n")
                        if hasattr(model, 'feature_importances_'):
                            buffer.write("Feature Importances:\n")
                            for feat, imp in zip(X.columns, model.feature_importances_):
                                buffer.write(f"{feat}: {imp:.4f}\n")
                        buffer.write("\n")
                    
                    st.download_button(
                        "📄 Download Model Info (TXT)",
                        buffer.getvalue(),
                        "model_details.txt",
                        "text/plain",
                        use_container_width=True
                    )
        
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            ### PharmaRelease AI v1.0
            **Developed by:** [Your Name]  
            **Contact:** your.email@example.com  
            
            This system uses machine learning to predict and optimize drug release profiles
            based on formulation parameters.
            
            © 2023 All Rights Reserved
            """)

if __name__ == "__main__":
    main()
