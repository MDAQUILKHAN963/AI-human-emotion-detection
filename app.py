import streamlit as st
import requests
from PIL import Image
import io

#st.title("Real-Time Web Facial Emotion Detection")

#st.write("Use your webcam to capture a photo and get an instant emotion prediction!")

#img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Display the image
    st.image(img_file_buffer, caption="Captured Image", use_column_width=True)
    # Prepare file for POST request
    image_data = img_file_buffer.getvalue()
    files = {'file': image_data}
    # Send to backend for prediction
    resp = requests.post("http://127.0.0.1:5000/predict", files=files)
    if resp.ok:
        st.success(f"Predicted Emotion: {resp.json()['emotion']}")
    else:
        st.error("Prediction failed. Please try again.")