import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# -------------------------------
# Load model
# -------------------------------
model = tf.keras.models.load_model("crop_disease_model.h5")

# -------------------------------
# Load class indices
# -------------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

# -------------------------------
# Treatments dictionary
# -------------------------------
treatments = {
    "Apple___Apple_scab": "Apply fungicide during early season and remove infected leaves.",
    "Apple___Black_rot": "Prune infected branches and apply appropriate fungicide.",
    "Apple___Cedar_apple_rust": "Use resistant varieties and apply fungicide in spring.",
    "Apple___healthy": "The apple crop is healthy. Maintain orchard hygiene.",

    "Blueberry___healthy": "The blueberry crop is healthy. Maintain proper care.",

    "Cherry_(including_sour)___healthy": "The cherry crop is healthy. Maintain proper nutrition.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply fungicide and improve air circulation.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use resistant hybrids and apply fungicide if required.",
    "Corn_(maize)___Common_rust_": "Apply fungicide and monitor crop regularly.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant varieties and practice crop rotation.",
    "Corn_(maize)___healthy": "The corn crop is healthy. Maintain proper fertilization.",

    "Grape___Black_rot": "Remove infected berries and apply suitable fungicide.",
    "Grape___Esca_(Black_Measles)": "Prune affected vines and avoid plant stress.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicide and maintain vineyard sanitation.",
    "Grape___healthy": "The grape crop is healthy. Maintain vineyard hygiene.",

    "Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees and control insect vectors.",

    "Peach___Bacterial_spot": "Apply bactericide sprays and use resistant varieties.",
    "Peach___healthy": "The peach crop is healthy. Maintain orchard care.",

    "Pepper,_bell___Bacterial_spot": "Use copper-based sprays and disease-free seeds.",
    "Pepper,_bell___healthy": "The pepper crop is healthy. Maintain proper irrigation.",

    "Potato___Early_blight": "Apply fungicide and practice crop rotation.",
    "Potato___Late_blight": "Use certified fungicide and remove infected plants.",
    "Potato___healthy": "The potato crop is healthy. Maintain proper care.",

    "Raspberry___healthy": "The raspberry crop is healthy. Maintain field hygiene.",
    "Soybean___healthy": "The soybean crop is healthy. Maintain proper fertilization.",

    "Squash___Powdery_mildew": "Apply fungicide and improve air circulation.",

    "Strawberry___healthy": "The strawberry crop is healthy. Maintain proper irrigation and hygiene.",
    "Strawberry___Leaf_scorch": "Remove infected leaves and apply fungicide if necessary.",

    "Tomato___Bacterial_spot": "Use copper-based bactericides and remove infected leaves.",
    "Tomato___Early_blight": "Remove infected leaves and apply recommended fungicide.",
    "Tomato___Late_blight": "Apply fungicide and destroy severely affected plants.",
    "Tomato___Leaf_Mold": "Improve air circulation and apply fungicide if needed.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves and apply suitable fungicide.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use miticides or neem oil and maintain field hygiene.",
    "Tomato___Target_Spot": "Apply fungicide and avoid overhead irrigation.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants and disinfect tools.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies and remove infected plants.",
    "Tomato___healthy": "The tomato crop is healthy. Maintain proper irrigation and nutrition."
}

# -------------------------------
# UI
# -------------------------------
st.title("🌱 Crop Disease Detection System")
st.write("Upload a crop leaf image to detect disease and view treatment suggestions.")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((128, 128))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    crop, disease = predicted_class.split("___")

    st.success(f"Crop: {crop}")
    st.success(f"Disease: {disease}")
    st.write(f"Confidence: {confidence:.2f}%")

    treatment = treatments.get(
        predicted_class,
        "Treatment information for this disease is not available."
    )

    st.info(f"Treatment: {treatment}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "⚠️ Disclaimer: AI-based predictions are for educational purposes only. "
    "Consult an agricultural expert before taking action."
)
