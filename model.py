import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title('wce curated colon')

model = tf.keras.models.load_model('D:\\New folder\\wcecolon.h5')


def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((100, 100))
    img = np.array(img) / 255.0 
    img = np.expand_dims(img, axis=0)
    return img
disease_mapping = {
    0: 'normal',
    1: 'ulcerative colitis',
    2: 'polyps',
    3: 'esophagitis'
}


def make_prediction(input_data):
    prediction = model.predict(input_data)
    prediction_result= np.argmax(prediction)
    disease_name=disease_mapping[prediction_result]
    return disease_name

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)
    if st.button('Make Prediction'):
        prediction_result = make_prediction(image)
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.success(f'Prediction:, {prediction_result}')
