import numpy as np
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import streamlit as st
from functions import load_img, create_model

# from keras import layers
# from keras import models
# from keras import optimizers
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras import Model

# from tensorflow.keras.initializers import glorot_uniform
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import Xception



PATH = "model_weights/"
WEIGHTS = 'xception_checkpoint.h5'
SHAPE = 300
CLASS_DICT = {
    0: 'bee',
    1: 'insect',
    2: 'other',
    3: 'wasp'
}

@st.cache(allow_output_mutation = True)
def load_own_model(weights_path):
    # return load_model(weights_path)
    model = create_model(shape = SHAPE)
    model.load_weights(weights_path)
    return model

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    result = st.empty()
    uploaded_img = st.file_uploader(label = 'Upload your image:')
    # uploaded_img = 'punture-vespe-1217.jpg'
    
    if uploaded_img:
        bytes_data = uploaded_img.read()
        st.image(bytes_data, caption = "your uploaded image: ", width = 350)
        result.info("Please wait for your results")
        model = load_own_model(PATH + WEIGHTS)
        pred_img = load_img(uploaded_img, SHAPE)
        pred = CLASS_DICT[np.argmax(model.predict(pred_img))]
        result.success("Result: " + pred)



