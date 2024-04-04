# AI-Activity
Code created using ML-AI techniques


For Handwritten Digit recognition MLP:

you need 10 images of digits of 28x28 to test (and copy to content folder)
You also need to create a web_app file in content folder, the code should be the following:
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('/content/hand_written_Digit_recog_model.keras')

st.header('Handwritten Digits Recognition Page')
img = st.text_input('Enter Image Name')
image = cv2.imread('/content/'+ img)[:,:,0]
image = np.invert(np.array([image]))

output = model.predict(image)
stn = 'Digit in the image is ' + str(np.argmax(output))
st.markdown(stn)
st.image(img, width = 150)

