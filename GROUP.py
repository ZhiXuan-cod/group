# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, accuracy_score
import shap

# -------------------------------
# Page configuration
st.set_page_config(page_title="AI Product Feature Prioritisation", layout="wide")
st.title("🚚 AI Product Feature Prioritisation Using Customer & IoT Data")

# -------------------------------
# Helper: Customer data preprocessing
@st.cache_data
def preprocess_customer(df):
    # Drop rows with missing target or numerical cols
    num_cols = ['DeliveryTimemin', 'CustomerServiceRating']
    target = 'Rating'
    df = df.dropna(subset=num_cols + [target])
    
    cat_cols = ['AgentName', 'Location', 'OrderType', 'CustomerFeedbackType',
                'PriceRange', 'ProductAvailability', 'OrderAccuracy', 'DiscountApplied']
    
    # One-hot encode categoricals
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Drop ReviewText (free text) and keep target
    X = df_enc.drop(columns=[target, 'ReviewText'], errors='ignore')
    y = df_enc[target].values
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler

# -------------------------------
# Helper: IoT data preprocessing
@st.cache_data
def preprocess_iot(df):
    # Keep only required features + target 'Logistics_Delay'
    # Use columns similar to notebook: Temperature, Humidity, Asset_Utilization, Waiting_Time
    # If Asset_Utilization missing, derive from Inventory_Level (example logic)
    if 'Asset_Utilization' not in df.columns:
        # Assume Inventory_Level as proxy (scaled)
        df['Asset_Utilization'] = df['Inventory_Level'] / 100.0 if 'Inventory_Level' in df.columns else 0.5
    if 'Waiting_Time' not in df.columns and 'Waiting' in df.columns:
        df['Waiting_Time'] = df['Waiting']
    elif 'Waiting_Time' not in df.columns:
        df['Waiting_Time'] = 0.0
    
    feature_cols = ['Temperature', 'Humidity', 'Asset_Utilization', 'Waiting_Time']
    # Drop rows with missing values in features or target
    df = df.dropna(subset=feature_cols + ['Logistics_Delay'])
    
    X = df[feature_cols].copy()
    y = df['Logistics_Delay'].astype(int)   # binary (0/1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, feature_cols, scaler

# -------------------------------
# Train regression model (Rating)
@st.cache_resource
def train_regression_model(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                        alpha=0.001, max_iter=200, early_stopping=True,
                        validation_fraction=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model

# -------------------------------
# Train classification model (Logistics Delay)
@st.cache_resource
def train_classification_model(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                        max_iter=200, early_stopping=True,
                        validation_fraction=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model

# -------------------------------
# Permutation importance (faster than SHAP for deployment)
def get_feature_importance(model, X_test, y_test, feature_names, task='regression'):
    scoring = 'neg_mean_absolute_error' if task == 'regression' else 'accuracy'
    result = permutation_importance(model, X_test, y_test, n_repeats=5,
                                    random_state=42, scoring=scoring)
    importance = result.importances_mean
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    return df_imp.sort_values('importance', ascending=False)

# -------------------------------
# Sidebar – file uploads
st.sidebar.header("📂 Upload CSV Files")
cust_file = st.sidebar.file_uploader("Customer Survey (Fast Delivery Agent Reviews.csv)", type="csv")
iot_file = st.sidebar.file_uploader("IoT Logistics Dataset (smart_logistics_dataset.csv)", type="csv")

if not cust_file or not iot_file:
    st.info("Please upload both CSV files to start the analysis.")
    st.stop()

# -------------------------------
# Load data
@st.cache_data
def load_customer(file):
    return pd.read_csv(file)

@st.cache_data
def load_iot(file):
    return pd.read_csv(file)

df_cust = load_customer(cust_file)
df_iot = load_iot(iot_file)

st.success("✅ Files loaded successfully!")

# -------------------------------
# CUSTOMER ANALYSIS (Product feature prioritisation)
st.header("📊 1. Customer Survey Analysis – Rating Prediction")

with st.spinner("Preprocessing customer data..."):
    X_tr_c, X_te_c, y_tr_c, y_te_c, cust_features, scaler_c = preprocess_customer(df_cust)

st.write(f"**Data shape** – Training: {X_tr_c.shape}, Test: {X_te_c.shape}")

if st.button("🚀 Train Regression Model (Rating)"):
    with st.spinner("Training neural network..."):
        reg_model = train_regression_model(X_tr_c, y_tr_c)
        y_pred = reg_model.predict(X_te_c)
        mae = mean_absolute_error(y_te_c, y_pred)
        st.metric("Test MAE (Rating)", f"{mae:.3f}")
    
    # Feature importance (permutation)
    with st.spinner("Computing feature importance..."):
        imp_df = get_feature_importance(reg_model, X_te_c, y_te_c, cust_features, task='regression')
    
    st.subheader("🏆 Feature Priority (Product Feature Prioritisation)")
    st.dataframe(imp_df, use_container_width=True)
    
    # Plot top 10
    fig, ax = plt.subplots(figsize=(10, 6))
    top10 = imp_df.head(10)
    sns.barplot(data=top10, x='importance', y='feature', palette='viridis', ax=ax)
    ax.set_title("Top 10 Features Influencing Customer Rating")
    ax.set_xlabel("Permutation Importance (Δ MAE)")
    st.pyplot(fig)
    
    # Download button
    csv = imp_df.to_csv(index=False)
    st.download_button("📥 Download Customer Feature Priority", csv, "customer_feature_priority.csv", "text/csv")
    
    # Optional SHAP summary (sample for explanation)
    if st.checkbox("Show SHAP summary (may take a few seconds)"):
        with st.spinner("Computing SHAP values..."):
            explainer = shap.KernelExplainer(reg_model.predict, X_tr_c[:100])
            shap_values = explainer.shap_values(X_te_c[:50])
            fig_shap = plt.figure()
            shap.summary_plot(shap_values, X_te_c[:50], feature_names=cust_features, show=False)
            st.pyplot(fig_shap)
    
    # --- Interactive prediction widget ---
    st.subheader("🔮 Predict Rating for a New Customer")
    with st.form("customer_form"):
        cols = st.columns(4)
        input_values = {}
        for i, feat in enumerate(cust_features):
            with cols[i % 4]:
                input_values[feat] = st.number_input(feat, value=0.0, step=0.1, format="%.2f")
        submitted = st.form_submit_button("Predict Rating")
        if submitted:
            input_df = pd.DataFrame([input_values])
            input_scaled = scaler_c.transform(input_df)
            pred_rating = reg_model.predict(input_scaled)[0]
            st.success(f"⭐ Predicted Rating: {pred_rating:.2f} / 5.0")

# -------------------------------
# IOT ANALYSIS
st.header("📡 2. IoT Logistics Analysis – Delay Prediction")

with st.spinner("Preprocessing IoT data..."):
    X_tr_i, X_te_i, y_tr_i, y_te_i, iot_features, scaler_i = preprocess_iot(df_iot)

st.write(f"**Data shape** – Training: {X_tr_i.shape}, Test: {X_te_i.shape}")

if st.button("🚀 Train Classification Model (Logistics Delay)"):
    with st.spinner("Training MLP classifier..."):
        clf_model = train_classification_model(X_tr_i, y_tr_i)
        y_pred_i = clf_model.predict(X_te_i)
        acc = accuracy_score(y_te_i, y_pred_i)
        st.metric("Test Accuracy (Delay Prediction)", f"{acc:.4f}")
    
    # Feature importance (Random Forest for comparison with original notebook)
    with st.spinner("Computing feature importance via Random Forest..."):
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_tr_i, y_tr_i)
        imp_rf = pd.DataFrame({'feature': iot_features, 'importance': rf.feature_importances_})
        imp_rf = imp_rf.sort_values('importance', ascending=False)
    
    st.subheader("🏆 IoT Feature Priority (Top factors causing delays)")
    st.dataframe(imp_rf, use_container_width=True)
    
    fig2, ax2 = plt.subplots()
    sns.barplot(data=imp_rf, x='importance', y='feature', palette='rocket', ax=ax2)
    ax2.set_title("IoT Feature Importance for Logistics Delay")
    st.pyplot(fig2)
    
    # Download
    csv_iot = imp_rf.to_csv(index=False)
    st.download_button("📥 Download IoT Feature Priority", csv_iot, "iot_feature_priority.csv", "text/csv")
    
    # --- Interactive prediction for IoT ---
    st.subheader("📦 Predict Delay Probability for a New Shipment")
    with st.form("iot_form"):
        i_cols = st.columns(4)
        i_vals = {}
        for i, feat in enumerate(iot_features):
            with i_cols[i % 4]:
                i_vals[feat] = st.number_input(feat, value=0.0, step=0.1, format="%.2f")
        i_submit = st.form_submit_button("Predict Delay")
        if i_submit:
            i_input = pd.DataFrame([i_vals])
            i_input_scaled = scaler_i.transform(i_input)
            delay_pred = clf_model.predict(i_input_scaled)[0]
            proba = clf_model.predict_proba(i_input_scaled)[0][1] if hasattr(clf_model, "predict_proba") else None
            if proba is not None:
                st.success(f"🚦 Delay predicted: {'YES' if delay_pred==1 else 'NO'} (probability: {proba:.2f})")
            else:
                st.success(f"🚦 Delay predicted: {'YES' if delay_pred==1 else 'NO'}")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit, scikit-learn, SHAP. Prioritise features that most impact customer rating and logistics delays.")