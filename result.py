import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Product Feature Prioritisation", layout="wide")
st.title("🚚 AI Product Feature Prioritisation Using Customer & IoT Data")

@st.cache_data
def load_customer_priority():
    try:
        return pd.read_csv("feature_priority.csv")
    except FileNotFoundError:
        st.error("feature_priority.csv not found. Please make sure the file is in the app directory.")
        return None

@st.cache_data
def load_iot_priority():
    try:
        return pd.read_csv("iot_feature_priority.csv")
    except FileNotFoundError:
        st.error("iot_feature_priority.csv not found. Please make sure the file is in the app directory.")
        return None

cust_imp = load_customer_priority()
iot_imp = load_iot_priority()

if cust_imp is None or iot_imp is None:
    st.stop()

# ------------------- Customer Section -------------------
st.header("📊 Customer Survey Analysis – Feature Priority")
st.dataframe(cust_imp, use_container_width=True)

fig1, ax1 = plt.subplots(figsize=(10, 6))
top10_cust = cust_imp.head(10)
sns.barplot(data=top10_cust, x='importance', y='feature', palette='viridis', ax=ax1)
ax1.set_title("Top 10 Features Influencing Customer Rating")
ax1.set_xlabel("Importance Score")
ax1.set_ylabel("Feature")
st.pyplot(fig1)

csv_cust = cust_imp.to_csv(index=False)
st.download_button(
    label="📥 Download Customer Feature Priority (CSV)",
    data=csv_cust,
    file_name="customer_feature_priority.csv",
    mime="text/csv"
)

# ------------------- IoT Section -------------------
st.header("📡 IoT Logistics Analysis – Delay Prediction")
st.dataframe(iot_imp, use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(data=iot_imp, x='importance', y='feature', palette='rocket', ax=ax2)
ax2.set_title("IoT Feature Importance for Logistics Delay")
ax2.set_xlabel("Importance Score")
ax2.set_ylabel("Feature")
st.pyplot(fig2)

csv_iot = iot_imp.to_csv(index=False)
st.download_button(
    label="📥 Download IoT Feature Priority (CSV)",
    data=csv_iot,
    file_name="iot_feature_priority.csv",
    mime="text/csv"
)

st.success("Analysis complete. Use the buttons above to download the results.")