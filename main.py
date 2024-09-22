import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set the working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "Rice_leaf_disease_detection.h5")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class indices
with open(os.path.join(working_dir, "class_indices.json")) as f:
    class_indices = json.load(f)

# Create a mapping from string index to class name
class_names = {int(k): v for k, v in class_indices.items()}  # Convert keys to int


# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image, target_size=(256, 256)):
    # Resize the image
    img = image.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    return img_array


# Function to predict the class of an image
def predict_image_class(model, image, class_names):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Get the predicted class name
    predicted_class_name = class_names.get(predicted_class_index, "Unknown")

    return predicted_class_name, predictions


# Streamlit App
st.title('Rice Disease Detection')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((256, 256))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction, raw_predictions = predict_image_class(model, image, class_names)
            st.success(f'Prediction: {str(prediction)}')
            st.write(f'Raw predictions: {raw_predictions}')
            st.write(f'Predicted class index: {np.argmax(raw_predictions, axis=1)[0]}')
            st.write(f'Class names: {class_names}')
