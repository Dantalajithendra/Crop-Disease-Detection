import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = tf.keras.models.load_model("crop_disease_model.h5")

# --------------------------------------------------
# Load class indices saved during training
# --------------------------------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> class name
class_names = {v: k for k, v in class_indices.items()}

# --------------------------------------------------
# Treatment information for each class
# --------------------------------------------------
treatments = {
    "Potato___Early_blight": "Apply recommended fungicide and avoid excessive moisture.",
    "Potato___healthy": "Crop is healthy. Maintain proper irrigation and nutrition.",

    "Tomato___Early_blight": "Remove infected leaves and use copper-based fungicide.",
    "Tomato___Late_blight": "Apply fungicide and destroy severely affected plants.",
    "Tomato___healthy": "Crop is healthy. Follow good agricultural practices."
}

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("🌱 Crop Disease Detection System")
st.write("Upload a crop leaf image to detect the disease and view treatment suggestions.")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((128, 128))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    # Display result
    st.success(f"Disease Detected: {predicted_class}")
    st.info(f"Treatment: {treatments[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ Disclaimer: This system provides AI-based predictions for educational purposes only. "
    "Please consult an agricultural expert before taking action."
)
