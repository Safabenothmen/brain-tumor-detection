import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import tempfile
import os

# Load the pre-trained model (UPDATED FOR .KERAS FORMAT)
model = tf.keras.models.load_model('brain_tumor_data_augmentation_model.keras')


# Class names remain the same
class_names = ["glioma", "meningioma", "no tumor", "pituitary"]

# Preprocess function (UNCHANGED)
def preprocess_image(image):
    # Resize to match model's expected sizing
    image = cv2.resize(image, (150, 150))
    # Normalize pixel values to be between 0 and 1
    image = image.astype('float32') / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app UI (UNCHANGED)
st.title("Brain Tumor Detection from MRI Scans")
st.write("Upload an MRI scan to classify the type of brain tumor or detect if no tumor is present")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # More efficient image handling (alternative to tempfile)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediction Results")
        st.metric("Diagnosis", class_names[predicted_class], 
                 f"{confidence:.1%} confidence")
        
    with col2:
        st.subheader("Confidence Breakdown")
        for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
            st.progress(float(prob), text=f"{class_name}: {prob:.2%}")