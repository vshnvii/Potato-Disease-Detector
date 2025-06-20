import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Load model
model = tf.keras.models.load_model("potato_model.keras")

# Load class names
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# Title
st.title("Potato Disease Classifier 🌿🥔")
st.write("Upload a potato leaf image and let the CNN predict the disease.")

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess (for models that do NOT include preprocessing layers)
    image = image.resize((256, 256))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    st.write(f"### Prediction: **{class_names[class_index]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

