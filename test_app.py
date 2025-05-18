import numpy as np
import tensorflow as tf
import os
import pytest

# Importe depuis app.py
from app import preprocess_image

# Test de la fonction de prétraitement
def test_preprocess_image():
    dummy_img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # image blanche
    processed = preprocess_image(dummy_img)

    assert processed.shape == (1, 150, 150, 3), "Mauvaise taille de sortie"
    assert processed.dtype == np.float32, "Mauvais type de données"
    assert processed.max() <= 1.0 and processed.min() >= 0.0, "Données non normalisées"

# Test du chargement du modèle
def test_model_load():
    assert os.path.exists('brain_tumor_data_augmentation_model.keras'), "Fichier modèle non trouvé"
    model = tf.keras.models.load_model('brain_tumor_data_augmentation_model.keras')
    assert model is not None, "Échec de chargement du modèle"

# Test d'une prédiction factice
def test_model_prediction():
    model = tf.keras.models.load_model('brain_tumor_data_augmentation_model.keras')
    dummy_input = np.random.rand(1, 150, 150, 3).astype(np.float32)
    prediction = model.predict(dummy_input)

    assert prediction.shape == (1, 4), "La sortie doit avoir 4 classes"
    assert np.all(prediction >= 0) and np.all(prediction <= 1), "Les probabilités doivent être entre 0 et 1"
