import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import requests

from gtts import gTTS
from googletrans import Translator

import google.generativeai as genai
import base64

#import plotly.express as px
import json 
import requests 
 

#import plotly.express as px 









# Function to translate text to Tamil
def translate_to_tamil(text):
    translator = Translator()
    translated = translator.translate(text, src='en', dest='ta')
    return translated.text

# Function to generate audio from Tamil text
def generate_audio(tamil_text):
    tts = gTTS(text=tamil_text, lang='ta')
    tts.save("output_translated_tamil.mp3")

# Configure Generative AI
genai.configure(api_key="AIzaSyBCmUwrvfwrTq2XDy1C0gFutl2O23p7oP4")
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array



def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name        

# Define generation configuration and safety settings
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

# Initialize GenerativeModel
model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Start conversation
convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["hi"]
  },
  {
    "role": "model",
    "parts": ["Hello, how can I assist you today?"]
  },
])

# Load plant disease prediction model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Streamlit App
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon=":herb:",
    layout="centered",  
    initial_sidebar_state="collapsed" 
)

st.sidebar.title('Chatbot')

# Placeholder for chat input
input_placeholder = st.sidebar.empty()

# Chat Input in sidebar
chat_input = input_placeholder.text_input("Type your Queries here:")

# Send button in sidebar
if st.sidebar.button('Chat'):
    convo.send_message(chat_input)
    fetched_remedies = convo.last.text
    st.session_state['fetched_remedies'] = fetched_remedies

# Display output from chatbot in sidebar
if st.session_state.get('fetched_remedies'):
    st.sidebar.subheader("Chatbot Response:")
    st.sidebar.write(st.session_state['fetched_remedies'])

# Initialize session state variable to store fetched remedies and prediction message
if 'fetched_remedies' not in st.session_state:
    st.session_state['fetched_remedies'] = ""
if 'prediction_message' not in st.session_state:
    st.session_state['prediction_message'] = ""

st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            prediction_message = f'Prediction: {str(prediction)}'
            st.success(prediction_message)
            st.session_state['prediction_message'] = prediction_message

            convo.send_message("Remedies for" + str(prediction) + ":")
            
            fetched_remedies = convo.last.text
            st.session_state['fetched_remedies'] = fetched_remedies
            #st.write(fetched_remedies)

if st.button('Get audio and Remedies'):
    fetched_remedies = st.session_state.get('fetched_remedies', "")
    prediction_message = st.session_state.get('prediction_message', "")

    if fetched_remedies:

        st.success(prediction_message)
        st.subheader("Remedies:")

        st.write(fetched_remedies.replace('*',' '))



        translated_text = translate_to_tamil(fetched_remedies)
        generate_audio(translated_text)
        st.success("Audio generated successfully!")
        audio_file = open('output_translated_tamil.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
         # Display prediction message again after translation
    else:
        st.warning("No text available to translate.")




if st.button('Disease info'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            prediction_message = f'Prediction: {str(prediction)}'
            
            st.session_state['prediction_message'] = prediction_message

            convo.send_message("Disease info and prevention" + str(prediction) + ":")
            
            fetched_remedies = convo.last.text
            st.session_state['fetched_remedies'] = fetched_remedies
            st.write(fetched_remedies)


if st.button('Get AntiBiotics'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            prediction_message = f'Prediction: {str(prediction)}'
            
            st.session_state['prediction_message'] = prediction_message

            convo.send_message("AntiBiotics for" + str(prediction) + ":")
            
            fetched_remedies = convo.last.text
            st.session_state['fetched_remedies'] = fetched_remedies
            st.write(fetched_remedies)