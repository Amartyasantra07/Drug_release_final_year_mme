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

# Configuration
st.set_page_config(
    page_title="Advanced Drug Release Prediction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    # Bagging Ensemble - Fixed parameter name to 'estimator'
    ensembles['Bagging'] = BaggingRegressor(
        estimator=_base_models['Decision Tree'],  # Changed from base_estimator
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
    st.sidebar.header("Model Configuration")
    
    model_categories = {
        "Linear Models": ["Linear", "Ridge", "Lasso", "Poly"],
        "Tree-based Models": ["Decision Tree", "Random Forest", "GBM"],
        "Boosted Trees": ["XGBoost", "LightGBM", "CatBoost"],
        "Other Models": ["SVR", "MLP"],
        "Ensemble Models": ["Stacking", "Bagging"]
    }
    
    selected_category = st.sidebar.selectbox(
        "Model Category",
        list(model_categories.keys())
    )
    
    model_option = st.sidebar.selectbox(
        "Select Model", 
        model_categories[selected_category]
    )
    
    # Model-specific parameters
    if model_option in ["Random Forest", "GBM", "XGBoost", "LightGBM", "CatBoost"]:
        n_estimators = st.sidebar.slider(
            "Number of estimators",
            50, 500, 
            value=100,
            step=50
        )
        max_depth = st.sidebar.slider("Max depth", 3, 10, 5)
        
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
        C = st.sidebar.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
        epsilon = st.sidebar.slider("Epsilon", 0.01, 1.0, 0.1, 0.01)
        all_models[model_option] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=C, epsilon=epsilon))
        ]).fit(X, y)
    
    elif model_option == "MLP":
        hidden_layers = st.sidebar.slider("Hidden layer size", 10, 200, 100)
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
    st.header("Input Parameters")
    cols = st.columns(3)
    inputs = {}
    
    with cols[0]:
        inputs['Time(min)'] = st.slider(
            "Time (min)", 
            float(X['Time(min)'].min()), 
            float(X['Time(min)'].max()),
            float(X['Time(min)'].median()),
            step=1.0
        )
        inputs['Drug_con(Mg)'] = st.slider(
            "Drug Concentration (Mg)", 
            float(X['Drug_con(Mg)'].min()), 
            float(X['Drug_con(Mg)'].max()),
            float(X['Drug_con(Mg)'].median()),
            step=0.1
        )
    
    with cols[1]:
        inputs['Rpm'] = st.slider(
            "RPM", 
            int(X['Rpm'].min()), 
            int(X['Rpm'].max()),
            int(X['Rpm'].median()),
            step=10
        )
        inputs['pH'] = st.slider(
            "pH", 
            float(X['pH'].min()), 
            float(X['pH'].max()),
            float(X['pH'].median()),
            step=0.1
        )
    
    with cols[2]:
        inputs['Temperature'] = st.slider(
            "Temperature", 
            int(X['Temperature'].min()), 
            int(X['Temperature'].max()),
            int(X['Temperature'].median()),
            step=1
        )
        st.markdown("---")
        st.markdown("### Quick Presets")
        preset_cols = st.columns(2)
        with preset_cols[0]:
            if st.button("Default Settings"):
                st.session_state.inputs = {k: float(X[k].median()) for k in X.columns}
        with preset_cols[1]:
            if st.button("Optimal (Max Release)"):
                st.session_state.inputs = {
                    'Time(min)': X['Time(min)'].max(),
                    'Drug_con(Mg)': X['Drug_con(Mg)'].median(),
                    'Rpm': X['Rpm'].max(),
                    'pH': X['pH'].median(),
                    'Temperature': X['Temperature'].median()
                }
    
    return pd.DataFrame([inputs])

def show_prediction_results(model_name, X_input, y_true=None):
    """Display prediction results with visual feedback"""
    
    # Make prediction
    start_time = time.time()
    model = all_models[model_name]
    prediction = model.predict(X_input)[0]
    pred_time = time.time() - start_time
    
    # Display results
    st.success(f"""
    ## Predicted Drug Release: {prediction:.2f}%  
    """)
    
    # Performance metrics
    if y_true is not None:
        y_pred = model.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metric_cols = st.columns(3)
        metric_cols[0].metric("R² Score", f"{r2:.3f}", "Model Fit")
        metric_cols[1].metric("MSE", f"{mse:.3f}", "Error")
        metric_cols[2].metric("Prediction Time", f"{pred_time:.4f}s", "Speed")
    
    # SHAP explanation
    if model_name in ["Decision Tree", "Random Forest", "XGBoost", "LightGBM"]:
        try:
            st.subheader("Feature Impact Analysis")
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            
            # Summary plot
            fig_shap, ax = plt.subplots()
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            st.pyplot(fig_shap)
            
            # Force plot for current prediction
            st.write("Impact on current prediction:")
            shap_input = pd.DataFrame(X_input, columns=X.columns)
            shap_values_single = explainer(shap_input)
            fig_force, ax = plt.subplots()
            shap.plots.waterfall(shap_values_single[0], max_display=10, show=False)
            st.pyplot(fig_force)
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {str(e)}")
    
    # Visual feedback
    if prediction < 30:
        st.warning("Low release predicted. Consider increasing time or RPM.")
    elif prediction > 80:
        st.info("High release predicted. May need to adjust for sustained release.")
    
    return prediction

def optimization_ui():
    st.header("Parameter Optimization")
    
    opt_target = st.radio(
        "Optimization Target",
        ["Maximize Drug Release", "Minimize Drug Release", "Target Release"],
        horizontal=True
    )
    
    target_value = None
    if opt_target == "Target Release":
        target_value = st.slider(
            "Target Release Percentage",
            float(y.min()), float(y.max()), float(y.mean()), 1.0
        )
    
    model_for_opt = st.selectbox(
        "Model for Optimization",
        ["XGBoost", "Random Forest", "Stacking"],
        index=0
    )
    
    if st.button("Run Optimization", type="primary"):
        with st.spinner("Finding optimal parameters using Bayesian optimization..."):
            # Define search space
            search_spaces = {
                'Time(min)': Real(X['Time(min)'].min(), X['Time(min)'].max()),
                'Drug_con(Mg)': Real(X['Drug_con(Mg)'].min(), X['Drug_con(Mg)'].max()),
                'Rpm': Integer(X['Rpm'].min(), X['Rpm'].max()),
                'pH': Real(X['pH'].min(), X['pH'].max()),
                'Temperature': Integer(X['Temperature'].min(), X['Temperature'].max())
            }
            
            # Define objective function
            def objective(params):
                input_df = pd.DataFrame([params])
                pred = all_models[model_for_opt].predict(input_df)[0]
                
                if "Maximize" in opt_target:
                    return -pred  # Minimize negative prediction
                elif "Minimize" in opt_target:
                    return pred
                else:
                    return abs(pred - target_value)
            
            # Run Bayesian optimization
            opt = BayesSearchCV(
                estimator=all_models[model_for_opt],
                search_spaces=search_spaces,
                n_iter=30,  # Reduced for demo purposes
                cv=3,
                n_jobs=-1,
                random_state=42
            )
            
            # Need to wrap in dummy estimator for BayesSearchCV
            from sklearn.base import BaseEstimator
            class DummyEstimator(BaseEstimator):
                def fit(self, X, y): return self
                def predict(self, X): return np.zeros(len(X))
            
            opt.estimator = DummyEstimator()
            opt.fit(X, y)
            
            # Get best parameters
            best_params = opt.best_params_
            best_value = -opt.best_score_ if "Maximize" in opt_target else opt.best_score_
            
            st.success("**Optimal Parameters Found**")
            
            # Display optimal parameters
            param_cols = st.columns(5)
            for i, (param, value) in enumerate(best_params.items()):
                param_cols[i].metric(
                    param,
                    f"{value:.2f}",
                    f"Range: {X[param].min():.1f}-{X[param].max():.1f}"
                )
            
            st.metric(
                "Predicted Drug Release", 
                f"{best_value:.2f}%",
                delta=f"Target: {target_value:.2f}%" if target_value else None
            )

def model_comparison_ui():
    st.header("Model Comparison")
    
    selected_models = st.multiselect(
        "Select models to compare",
        list(all_models.keys()),
        default=["Linear", "Random Forest", "XGBoost", "Stacking"]
    )
    
    if st.button("Compare Models"):
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
        st.dataframe(
            comparison_df.sort_values('R²', ascending=False),
            hide_index=True,
            use_container_width=True
        )
        
        # Visualization
        fig = px.bar(
            comparison_df,
            x='Model',
            y='R²',
            error_y='CV R² Std',
            title='Model Performance Comparison (R² Score)',
            color='Model'
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/150x50?text=Pharma+Logo", use_column_width=True)
    model_option = model_selection_ui()
    
    # Main content
    st.title("💊 Advanced Drug Release Prediction System")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Prediction", 
        "📊 Analysis", 
        "⚡ Optimization", 
        "📊 Model Comparison",
        "⚙️ Settings"
    ])
    
    with tab1:
        input_df = input_parameters_ui()
        if st.button("Predict", type="primary"):
            with st.spinner("Making prediction..."):
                prediction = show_prediction_results(model_option, input_df, y)
                
                # Show historical similar runs
                st.subheader("Similar Historical Runs")
                df['similarity'] = np.sqrt(
                    ((df[['Time(min)', 'Drug_con(Mg)', 'Rpm', 'pH', 'Temperature']] - input_df.values) ** 2).sum(axis=1))
                similar_runs = df.nsmallest(3, 'similarity')
                st.dataframe(similar_runs.drop(columns=['similarity']))
    
    with tab2:
        st.header("Data Analysis")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Parameter Relationships", "Model Performance", "Data Distribution"]
        )
        
        if analysis_type == "Parameter Relationships":
            x_axis = st.selectbox("X-axis", X.columns)
            y_axis = st.selectbox("Y-axis", ['Drug_Release'] + list(X.columns))
            
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
                title="Actual vs Predicted Drug Release"
            )
            fig.add_shape(
                type="line", 
                x0=y.min(), y0=y.min(), 
                x1=y.max(), y1=y.max(),
                line=dict(color="Red", width=2, dash="dash")
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        optimization_ui()
    
    with tab4:
        model_comparison_ui()
    
    with tab5:
        st.header("System Settings")
        
        st.subheader("Data Management")
        if st.button("Reload Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.subheader("Export Results")
        export_type = st.selectbox(
            "Export Type",
            ["Current Predictions", "Model Details", "Optimization Results"]
        )
        
        if st.button("Generate Export"):
            if export_type == "Current Predictions":
                y_pred = all_models[model_option].predict(X)
                results = df.copy()
                results['Predicted'] = y_pred
                csv = results.to_csv(index=False)
                
                st.download_button(
                    "Download Predictions",
                    csv,
                    "drug_release_predictions.csv",
                    "text/csv"
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
                    "Download Model Info",
                    buffer.getvalue(),
                    "model_details.txt",
                    "text/plain"
                )

if __name__ == "__main__":
    main()