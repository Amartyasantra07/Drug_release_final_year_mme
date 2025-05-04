import streamlit as st

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import joblib

import matplotlib.pyplot as plt
import plotly.express as px
from io import StringIO
import time


# Set page config FIRST
st.set_page_config(layout="wide", page_icon="💊💊", page_title="🧪 Advanced Drug Release Prediction", initial_sidebar_state="expanded")

# Load the dataset
df = pd.read_csv('taguchi1.csv')

# Prepare the data
X = df.drop(columns=['Run ', 'perc of Drug Release'])
y = df['perc of Drug Release']

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Data augmentation using polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Dictionary of models with their parameter grids for tuning
MODELS = {
    "Enhanced MLR": {
        "model": LinearRegression(),
        "params": {'fit_intercept': [True, False]}
    },
    "Ridge Regression": {
        "model": Ridge(),
        "params": {'alpha': [0.1, 1.0, 10.0], 'fit_intercept': [True, False]}
    },
    "Lasso Regression": {
        "model": Lasso(),
        "params": {'alpha': [0.1, 1.0, 10.0], 'fit_intercept': [True, False]}
    },
    "ElasticNet": {
        "model": ElasticNet(),
        "params": {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}
    },
    "Support Vector Regressor": {
        "model": SVR(),
        "params": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    },
    "k-Nearest Neighbors": {
        "model": KNeighborsRegressor(),
        "params": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    },
    "Decision Tree": {
        "model": DecisionTreeRegressor(),
        "params": {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
    },
    "Random Forest": {
        "model": RandomForestRegressor(),
        "params": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, None]}
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(),
        "params": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    },
    "XGBoost": {
        "model": XGBRegressor(),
        "params": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    },
    "LightGBM": {
        "model": LGBMRegressor(),
        "params": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    },
    "AdaBoost": {
        "model": AdaBoostRegressor(),
        "params": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    },
    "Neural Network": {
        "model": MLPRegressor(max_iter=1000),
        "params": {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh']}
    },
    "Gaussian Process": {
        "model": GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0)),
        "params": {}
    }
}

# Function to train and evaluate a model# Function to train and evaluate a model
def train_model(model_name, X_train, y_train):
    model_info = MODELS[model_name]
    model = model_info["model"]
    params = model_info["params"]

    if params:  # Perform hyperparameter tuning if parameters are specified
        grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        model.fit(X_train, y_train)
        best_model = model
        best_params = "Default Parameters"

    return best_model, best_params


# Streamlit UI
def main():
    #st.title("Drug Prediction App")
    #st.title("💊💊 Advanced Drug Release Prediction and Optimization System💊💊")
    st.markdown("💊💊 Welcome to Drug Release Prediction")
    st.write("""
    This application predicts drug release percentage based on formulation parameters
    and recommends optimal conditions for maximum/minimum release using various machine learning algorithms.
    """)

    # Add specifications section
    st.sidebar.title("⚙️ System Specifications")
    st.sidebar.write("""
    - **Input Parameters**: Time(min), Drug Concentration(Mg), RPM, pH, Temperature
    - **Output**: Drug Release Percentage (%)
    - **Algorithms Available**: 14 different ML models
    - **Features**: Prediction, Optimization, Visualization, Export
    - **Data Points**: 27 experimental runs
    - **Validation**: 5-fold cross-validation
    """)

    # Model selection with description
    model_option = st.selectbox(
        "Select Prediction Model",
        list(MODELS.keys()),
        help="Select from various regression algorithms"
    )

    # Model description
    model_descriptions = {
        "Enhanced MLR": "Standard linear regression with polynomial features",
        "Ridge Regression": "Linear regression with L2 regularization",
        "Lasso Regression": "Linear regression with L1 regularization (feature selection)",
        "ElasticNet": "Linear regression with combined L1 and L2 regularization",
        "Support Vector Regressor": "Powerful for high-dimensional data with kernel tricks",
        "k-Nearest Neighbors": "Instance-based learning using nearby points",
        "Decision Tree": "Non-linear model with interpretable tree structure",
        "Random Forest": "Ensemble of decision trees with better generalization",
        "Gradient Boosting": "Sequential building of trees to correct errors",
        "XGBoost": "Optimized gradient boosting with regularization",
        "LightGBM": "Fast gradient boosting framework by Microsoft",
        "AdaBoost": "Adaptive boosting by focusing on hard samples",
        "Neural Network": "Multi-layer perceptron for complex patterns",
        "Gaussian Process": "Probabilistic model with uncertainty estimates"
    }

    st.write(f"**{model_option}**: {model_descriptions[model_option]}")

    # Train or load model
    @st.cache_resource
    def get_model_and_params(model_name):
        return train_model(model_name, X_poly, y)

    with st.spinner(f"Training {model_option} model..."):
        model, best_params = get_model_and_params(model_option)

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📈 Visualization", "⚡ Optimization", "📤 Export"])

    with tab1:
        st.header("Drug Release Prediction")
        col1, col2 = st.columns(2)

        with col1:
            # User inputs with default values set to medians
            input_values = []
            for col in X.columns:
                median_val = float(X[col].median())
                value = st.number_input(
                    f"{col}",
                    min_value=float(X[col].min()),
                    max_value=float(X[col].max()),
                    value=median_val,
                    step=1.0 if col in ['Rpm', 'Temperature'] else 0.1,
                    key=f"input_{col}"
                )
                input_values.append(value)

        if st.button("Predict Drug Release"):
            input_values = np.array(input_values).reshape(1, -1)
            input_values_scaled = scaler.transform(input_values)
            input_values_poly = poly.transform(input_values_scaled)
            prediction = model.predict(input_values_poly)[0]

            # Display prediction with confidence interval
            y_pred = model.predict(X_poly)
            residuals = y - y_pred
            std_dev = np.std(residuals)

            st.success(f"""
            **Predicted Drug Release:** {prediction:.2f}%
            **Confidence Interval (95%):** {prediction-1.96*std_dev:.2f}% to {prediction+1.96*std_dev:.2f}%
            """)

            # Model evaluation
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')

            st.subheader("Model Performance")
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("R² Score", f"{r2:.3f}", help="Higher is better (1.0 is perfect)")
            with col_metric2:
                st.metric("MSE", f"{mse:.3f}", help="Lower is better")
            with col_metric3:
                st.metric("CV MSE", f"{-np.mean(cv_scores):.3f}", help="Cross-validated MSE")

            # Feature importance (for models that support it)
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                st.subheader("Feature Importance")
                features = poly.get_feature_names_out(X.columns)

                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_)

                # Create dataframe for better display
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)

                st.dataframe(importance_df.head(10))  # Show top 10 features

                fig_imp = px.bar(importance_df.head(10), x='Feature', y='Importance',
                                 title='Top 10 Important Features')
                st.plotly_chart(fig_imp)

    with tab2:
        st.header("Data Visualization")

        # 3D Scatter plot
        st.subheader("3D Parameter Space Exploration")
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            x_axis = st.selectbox("X-axis", X.columns, index=0, key='x_axis')
        with col_y:
            y_axis = st.selectbox("Y-axis", X.columns, index=1, key='y_axis')
        with col_z:
            z_axis = st.selectbox("Z-axis (color)", ['perc of Drug Release'] + list(X.columns), index=0, key='z_axis')

        fig_3d = px.scatter_3d(df, x=x_axis, y=y_axis, z='perc of Drug Release',
                                color=z_axis, hover_name='Run ',
                                title=f"3D View: {x_axis} vs {y_axis} vs Drug Release")
        st.plotly_chart(fig_3d)

        # Actual vs Predicted plot
        st.subheader("Model Performance Visualization")
        y_pred = model.predict(X_poly)
        fig_perf = px.scatter(x=y, y=y_pred,
                                    labels={'x': 'Actual Drug Release', 'y': 'Predicted Drug Release'},
                                    title="Actual vs Predicted Drug Release")
        fig_perf.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                            line=dict(color="Red", width=2, dash="dash"))
        st.plotly_chart(fig_perf)

        # Residual plot
        st.subheader("Residual Analysis")
        residuals = y - y_pred
        fig_resid = px.scatter(x=y_pred, y=residuals,
                                     labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                     title="Residuals vs Predicted Values")
        fig_resid.add_hline(y=0, line_dash="dot", line_color="red")
        st.plotly_chart(fig_resid)

    with tab3:
        st.header("Optimal Parameter Recommendation")
        st.write("Find parameters that maximize or minimize drug release based on the model.")

        target = st.radio("Optimization Target", ["Maximize Drug Release", "Minimize Drug Release"])
        n_iterations = st.slider("Number of optimization iterations", 100, 10000, 1000, step=100)

        if st.button("Find Optimal Parameters"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Generate random samples within parameter bounds
            random_samples = pd.DataFrame()
            for col in X.columns:
                random_samples[col] = np.random.uniform(
                    X[col].min(), X[col].max(), n_iterations
                )

            best_value = -np.inf if "Maximize" in target else np.inf
            best_params = None

            for i in range(n_iterations):
                sample = random_samples.iloc[i:i+1]
                sample_scaled = scaler.transform(sample)
                sample_poly = poly.transform(sample_scaled)
                pred = model.predict(sample_poly)[0]

                if ("Maximize" in target and pred > best_value) or ("Minimize" in target and pred < best_value):
                    best_value = pred
                    best_params = sample.iloc[0]

                # Update progress
                progress = (i + 1) / n_iterations
                progress_bar.progress(progress)
                status_text.text(f"Iteration {i+1}/{n_iterations} | Current Best: {best_value:.2f}%")

            progress_bar.empty()
            status_text.empty()

            st.success(f"**Optimal Parameters for {target}:**")
            st.write(best_params)
            st.metric("Predicted Drug Release", f"{best_value:.2f}%")

            # Show how these compare to original data
            st.subheader("Comparison with Experimental Data")
            if "Maximize" in target:
                exp_max = df.loc[df['perc of Drug Release'].idxmax()]
                st.write("Experimental Maximum:", exp_max)
            else:
                exp_min = df.loc[df['perc of Drug Release'].idxmin()]
                st.write("Experimental Minimum:", exp_min)

    with tab4:
        st.header("Export Results")
        st.write("Download predictions or model details.")

        # Export predictions
        y_pred = model.predict(X_poly)
        results_df = df.copy()
        results_df['Predicted Release'] = y_pred
        results_df['Residual'] = y - y_pred

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv,
            file_name='drug_release_predictions.csv',
            mime='text/csv'
        )

        # Export model details
        if st.button("Export Model Summary"):
            buffer = StringIO()
            buffer.write(f"Drug Release Prediction Model Summary\n")
            buffer.write=("*50 + \n")
            buffer.write(f"Model Type: {model_option}\n")
            buffer.write(f"R² Score: {r2_score(y, y_pred):.4f}\n")
            buffer.write(f"MSE: {mean_squared_error(y, y_pred):.4f}\n\n")

            buffer.write("Best Hyperparameters:\n")
            if isinstance(best_params, dict):
                for param, value in best_params.items():
                    buffer.write(f"{param}: {value}\n")
            else:
                buffer.write("Default Parameters\n")

            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                buffer.write("\nFeature Importance:\n")
                features = poly.get_feature_names_out(X.columns)

                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_)

                for feat, imp in zip(features, importance):
                    buffer.write(f"{feat}: {imp:.4f}\n")

            st.download_button(
                label="Download Model Summary",
                data=buffer.getvalue(),
                file_name='model_summary.txt',
                mime='text/plain'
            )


if __name__ == "__main__":
    main()
