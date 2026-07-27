import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation App", page_icon="📊", layout="centered"
)

st.title("📊 Customer Segmentation Web App")
st.write("Customer के **RFM (Amount, Frequency, Recency)** मान दर्ज करें:")


# Model और Scaler लोड करने का फ़ंक्शन
@st.cache_resource
def load_artifacts():
  with open("kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)
  with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
  return model, scaler


kmeans, scaler = load_artifacts()

# User Inputs (Form Fields)
st.subheader("📥 Customer Metrics Input:")
amount = st.number_input(
    "Monetary Value (Amount Spent in ₹)", min_value=0.0, value=1500.0, step=100.0
)
frequency = st.number_input(
    "Frequency (Total Number of Orders)", min_value=1, value=10, step=1
)
recency = st.number_input(
    "Recency (Days since last purchase)", min_value=0, value=30, step=1
)

# Prediction Button
if st.button("🚀 Predict Customer Segment"):
  # Input Array बनाना
  raw_input = np.array([[amount, frequency, recency]])

  # Standard Scaler से Transform करना
  scaled_input = scaler.transform(raw_input)

  # Cluster Prediction
  cluster_id = kmeans.predict(scaled_input)[0]

  st.markdown("---")
  st.subheader(f"🎯 Assigned Cluster ID: **Cluster {cluster_id}**")

  # Cluster Mapping & Insights Display
  if cluster_id == 0:
    st.success(
        "🌟 **VIP / High-Value Customer!**\n"
        "- **विशेषता:** उच्च खर्च, बार-बार खरीदारी, हाल ही में खरीदारी की।"
    )
  elif cluster_id == 1:
    st.info(
        "👍 **Regular / Moderate Customer**\n"
        "- **विशेषता:** औसत खर्च और औसत खरीदारी आवृति।"
    )
  else:
    st.warning(
        "⚠️ **At-Risk / Low-Value Customer**\n"
        "- **विशेषता:** कम खर्च, बहुत समय से खरीदारी नहीं की।"
    )
