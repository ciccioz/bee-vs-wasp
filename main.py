# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import streamlit as st
from tensorflow.keras import layers
# from functions import load_img


def load_img(input_image, shape):
    img = Image.open(input_image).convert('RGB')
    img = img.resize((shape, shape))
    img = image.img_to_array(img)
    return np.reshape(img, [1, shape, shape, 3])/255


PATH = "model_weights/"
WEIGHTS = "xception.h5"
CLASS_DICT = {
    0: 'bee',
    1: 'wasp',
    2: 'insect',
    3: 'other'
}

@st.cache(allow_output_mutation = True)
def load_own_model(weights):
    return load_model(weights)

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    result = st.empty()
    uploaded_img = st.file_uploader(label = 'upload your image:')
    result.info("ciao")

    if uploaded_img:
        st.image(uploaded_img, caption = "your uploaded image: ", width=350)
        result.info("Please wait for your results")
        model = load_own_model(PATH + WEIGHTS)
        pred_img = load_img(uploaded_img, 224)
        pred = CLASS_DICT[np.argmax(model.predict(pred_img))]
        result.success("The breed of durian is " + pred)



