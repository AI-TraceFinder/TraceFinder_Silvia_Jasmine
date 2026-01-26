import streamlit as st
import numpy as np
from PIL import Image
import random

# ------------------------
# Page Config
# ------------------------
st.set_page_config(
    page_title="TraceFinder - Forensic Scanner Identification",
    layout="wide"
)

# ------------------------
# Dummy Class Setup (TEMP)
# ------------------------
class_names = ['Canon', 'Epson', 'HP', 'Xerox']

scanner_models = {
    "HP": "HP ScanJet Pro 2500",
    "Canon": "Canon DR-C240",
    "Epson": "Epson Perfection V39",
    "Xerox": "Xerox DocuMate 3125"
}

IMG_SIZE = 224

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("📁 Upload Scanner Image")
uploaded_file = st.sidebar.file_uploader(
    "Upload Image", type=["jpg", "png", "jpeg"]
)

# ------------------------
# Main Title
# ------------------------
st.title("🔍 TraceFinder – Forensic Scanner Identification Dashboard")
st.markdown("---")

# ------------------------
# Dummy Prediction Function
# ------------------------
def predict_image(image):
    predicted_class = random.choice(class_names)
    confidence = random.uniform(65, 95)
    return predicted_class, confidence


def confidence_level(conf):
    if conf > 90:
        return "Very High"
    elif conf > 70:
        return "High"
    elif conf > 50:
        return "Medium"
    else:
        return "Low"


# ------------------------
# Display Results
# ------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        scanner_brand, confidence = predict_image(image)
        level = confidence_level(confidence)
        scanner_model = scanner_models.get(scanner_brand, "Unknown Model")

        st.subheader("🖨️ Scanner Identification")
        st.success(f"**Scanner Brand:** {scanner_brand}")
        st.info(f"**Scanner Model:** {scanner_model}")
        st.metric("Confidence Score", f"{confidence:.2f}%")
        st.warning(f"Confidence Level: {level}")

    st.markdown("---")

    # Feature Analysis
    st.subheader("📊 Feature Analysis")
    st.write("**Detection Method:** AI Vision Simulation (Demo Mode)")

    # Forensic Details
    st.subheader("🧪 Forensic Details")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.success("### ✅ Primary Indicators")
        st.write("Scanner detected using simulated classifier.")

    with f2:
        st.info("### 📌 Secondary Indicators")
        st.write("Confidence generated for UI validation.")

    with f3:
        st.warning("### 🚨 Anomalies")
        if confidence > 70:
            st.write("No anomalies detected – clean scan.")
        else:
            st.write("Low confidence – manual verification recommended.")

else:
    st.info("👈 Please upload a scanner image from the sidebar to start analysis.")
