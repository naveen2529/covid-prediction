import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# ================================
# Load your trained model
# ================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("covid_vgg16_model.h5")
    return model

model = load_model()

# Class labels (change if your dataset had different classes)
class_labels = ['Covid', 'Normal', 'Viral Pneumonia']  

# ================================
# Streamlit UI
# ================================
st.title("🩺 COVID-19 X-Ray Classifier")
st.write("Upload a chest X-ray image to check if it is **COVID-19, Normal, or Viral Pneumonia**")

# File uploader
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Show results
    st.subheader("🔍 Prediction Result")
    st.write(f"**Class:** {class_labels[pred_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Probability chart
    st.bar_chart(dict(zip(class_labels, prediction[0])))
