
from gtts import gTTS
import os
from googletrans import Translator

# Define the English text to be translated
english_text = "please help me"

# Translate the English text to Tamil
translator = Translator()
translated = translator.translate(english_text, src='en', dest='ta')

# Get the translated Tamil text
tamil_text = translated.text

# Create a gTTS object with Tamil language code 'ta'
tts = gTTS(text=tamil_text, lang='ta')

# Save the speech to a file
tts.save("output_translated_tamil.mp3")

# Play the speech
os.system("start output_translated_tamil.mp3")
