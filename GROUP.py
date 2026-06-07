import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import shap
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, when, col
from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import tempfile
import os

# ------------------------------------------------------------
# Page config
st.set_page_config(page_title="AI Product Feature Prioritisation", layout="wide")
st.title("🚚 AI Product Feature Prioritisation (Exact Colab Replica)")

# ------------------------------------------------------------
# Helper: Cache data loading
@st.cache_data
def load_customer(file):
    return pd.read_csv(file)

@st.cache_data
def load_iot(file):
    return pd.read_csv(file)

# ------------------------------------------------------------
# Sidebar uploads
st.sidebar.header("📂 Upload CSV Files")
cust_file = st.sidebar.file_uploader("Customer Survey (Fast Delivery Agent Reviews.csv)", type="csv")
iot_file = st.sidebar.file_uploader("IoT Dataset (smart_logistics_dataset.csv)", type="csv")

if not cust_file or not iot_file:
    st.info("Please upload both CSV files to start the analysis.")
    st.stop()

# ------------------------------------------------------------
# Load data
df_cust = load_customer(cust_file)
df_iot = load_iot(iot_file)
st.success("✅ Files loaded successfully!")

# ------------------------------------------------------------
# 1. CUSTOMER ANALYSIS (TensorFlow DNN + SHAP) – exactly as Colab
st.header("📊 1. Customer Survey Analysis – Rating Prediction (TensorFlow DNN)")

# Preprocessing (identical to Colab)
cat_cols = ['AgentName', 'Location', 'OrderType', 'CustomerFeedbackType', 
            'PriceRange', 'ProductAvailability', 'OrderAccuracy', 'DiscountApplied']
num_cols = ['DeliveryTimemin', 'CustomerServiceRating']
target = 'Rating'

df_cust = df_cust.dropna(subset=num_cols + [target])
df_enc = pd.get_dummies(df_cust, columns=cat_cols, drop_first=True)
X = df_enc.drop(columns=[target, 'ReviewText'], errors='ignore')
y = df_enc[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

feature_names = X.columns.tolist()

# Train DNN (exact architecture from Colab)
@st.cache_resource
def train_dnn(X_train_s, y_train):
    model = Sequential([
        Input(shape=(X_train_s.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_s, y_train, validation_split=0.2, epochs=100, batch_size=32,
              callbacks=[early_stop], verbose=0)
    return model

if st.button("🚀 Train Rating Model (TensorFlow DNN)"):
    with st.spinner("Training Deep Neural Network..."):
        model = train_dnn(X_train_s, y_train)
        test_loss, test_mae = model.evaluate(X_test_s, y_test, verbose=0)
        st.metric("Test MAE (Rating)", f"{test_mae:.3f}")
        st.caption(f"(Expected ~1.022 as in Colab)")

    # SHAP feature importance (identical to Colab)
    with st.spinner("Computing SHAP feature importance..."):
        background = X_train_s[np.random.choice(X_train_s.shape[0], 100, replace=False)]
        explainer = shap.Explainer(model, background)
        shap_values = explainer(X_test_s[:100])
        mean_shap = np.abs(shap_values.values).mean(axis=0)
        feat_imp = pd.DataFrame({'feature': feature_names, 'importance': mean_shap})
        feat_imp = feat_imp.sort_values('importance', ascending=False)

    st.subheader("🏆 Feature Priority (Product Prioritisation)")
    st.dataframe(feat_imp, use_container_width=True)

    # Plot top 10
    fig, ax = plt.subplots(figsize=(10,6))
    top10 = feat_imp.head(10)
    sns.barplot(data=top10, x='importance', y='feature', palette='viridis', ax=ax)
    ax.set_title("Top 10 Features Influencing Customer Rating (SHAP)")
    st.pyplot(fig)

    # Download CSV
    csv_data = feat_imp.to_csv(index=False)
    st.download_button("📥 Download Customer Feature Priority", csv_data,
                       "feature_priority.csv", "text/csv")

# ------------------------------------------------------------
# 2. IoT ANALYSIS (PySpark – exactly as Colab)
st.header("📡 2. IoT Logistics Analysis – Delay Prediction (PySpark MLP + RF)")

# Need to save uploaded file to disk for Spark to read (Spark cannot read from memory)
@st.cache_data
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

iot_path = save_uploaded_file(iot_file)

# Initialize Spark session (with proper configuration for Streamlit)
@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("IoT_logistics") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

spark = get_spark_session()

if st.button("🚀 Run IoT Analysis (PySpark DNN + RandomForest)"):
    with st.spinner("Loading IoT data with Spark..."):
        df_iot_spark = spark.read.option("header", True).option("inferSchema", True).csv(iot_path)
        st.write(f"Rows: {df_iot_spark.count()}, Columns: {len(df_iot_spark.columns)}")

        # Show raw data preview
        st.subheader("Raw IoT Data Preview")
        st.dataframe(df_iot_spark.limit(5).toPandas())

        # --- IoT Analysis (exact from Colab) ---
        # Temperature bin analysis
        temp_delay = df_iot_spark.withColumn(
            "TempBin",
            when(col("Temperature") < 20, "20C")
            .when((col("Temperature") >= 20) & (col("Temperature") < 30), "20-30C")
            .otherwise("30C")
        ).groupBy("TempBin").agg(avg("Logistics_Delay").alias("delay_rate"))
        
        # Traffic impact
        traffic_impact = df_iot_spark.groupBy("Traffic_Status").agg(
            avg("Logistics_Delay").alias("delay_rate"),
            count("*").alias("count")
        )

        st.subheader("Analysis: Delay Rate by Temperature Bin")
        st.dataframe(temp_delay.toPandas())
        st.subheader("Analysis: Delay Rate by Traffic Status")
        st.dataframe(traffic_impact.toPandas())

        # Feature Engineering for ML
        feature_cols = ['Temperature', 'Humidity', 'Asset_Utilization', 'Waiting_Time']
        # Ensure Waiting_Time column exists (use 'Waiting' if present)
        if 'Waiting_Time' not in df_iot_spark.columns and 'Waiting' in df_iot_spark.columns:
            df_iot_spark = df_iot_spark.withColumnRenamed('Waiting', 'Waiting_Time')
        
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
        df_assembled = assembler.transform(df_iot_spark).select("features_raw", "Logistics_Delay")
        scaler_ml = SparkScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
        df_scaled = scaler_ml.fit(df_assembled).transform(df_assembled)
        train, test = df_scaled.randomSplit([0.8, 0.2], seed=42)

        # DNN (MLP) on IoT data – same layers as Colab: [4, 8, 5, 2]
        mlp = MultilayerPerceptronClassifier(
            layers=[len(feature_cols), 8, 5, 2],
            labelCol="Logistics_Delay",
            featuresCol="features",
            maxIter=50,
            seed=123
        )
        with st.spinner("Training MLP classifier on IoT data..."):
            mlp_model = mlp.fit(train)
            pred = mlp_model.transform(test)
            evaluator = MulticlassClassificationEvaluator(labelCol="Logistics_Delay", metricName="accuracy")
            acc = evaluator.evaluate(pred)
            st.metric("IoT DNN Accuracy", f"{acc:.4f}")
            st.caption(f"(Expected ~0.5617 as in Colab)")

        # Feature Importance via RandomForest (same as Colab)
        rf = RandomForestClassifier(labelCol="Logistics_Delay", featuresCol="features", numTrees=50)
        rf_model = rf.fit(train)
        imp = rf_model.featureImportances.toArray()
        iot_imp = pd.DataFrame({'feature': feature_cols, 'importance': imp})
        iot_imp = iot_imp.sort_values('importance', ascending=False)

        st.subheader("🏆 IoT Feature Priority (RandomForest)")
        st.dataframe(iot_imp, use_container_width=True)

        # Plot
        fig2, ax2 = plt.subplots()
        sns.barplot(data=iot_imp, x='importance', y='feature', palette='rocket', ax=ax2)
        ax2.set_title("IoT Feature Importance for Logistics Delay")
        st.pyplot(fig2)

        # Download
        csv_iot = iot_imp.to_csv(index=False)
        st.download_button("📥 Download IoT Feature Priority", csv_iot,
                           "iot_feature_priority.csv", "text/csv")

    # Stop Spark session to free resources (optional)
    spark.stop()
    st.success("Analysis complete! You can now download the CSVs.")

# Cleanup temp file
if os.path.exists(iot_path):
    os.unlink(iot_path)