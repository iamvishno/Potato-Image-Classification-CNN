import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.h5')
classnames = ['Early Blight', 'Late Blight', 'Healthy']
# Streamlit UI
st.title("Image Classification Web App")
st.write("Upload an image and the model will predict the class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=False, width=300)


    img_resized = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(img_resized)
    predicted_class = np.argmax(prediction, axis=1)

    st.write(f"## Prediction: {classnames[0]}")

