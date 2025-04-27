import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Charger le modèle
model = tf.keras.models.load_model('brain_tumor_data_augmentation_model.h5')

# Interface
st.title('Détection de Tumeur Cérébrale 🧠')
st.write('Uploadez une image IRM pour détecter la présence d\'une tumeur.')

uploaded_file = st.file_uploader("Choisissez une image IRM...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Image IRM téléchargée.', use_column_width=True)
    
    # Prétraitement
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    classes = ['glioma', 'meningioma', 'pituitary', 'no tumor']

    result = classes[predicted_class]

    if result == 'no tumor':
        st.success('✅ Aucun signe de tumeur détecté.')
    else:
        st.error(f'⚠️ Tumeur détectée : {result.capitalize()}')
