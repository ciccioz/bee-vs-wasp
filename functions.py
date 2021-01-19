
# function to convert uploaded images to the format for our model to consume.
# We specifically use the Image class from PIL because the uploaded image is in BytesIO format.
def load_img(input_image, shape):
    import numpy as np
    from PIL import Image
    from tensorflow.keras.preprocessing import image
    
    img = Image.open(input_image).convert('RGB')
    img = img.resize((shape, shape))
    img = image.img_to_array(img)
    return np.reshape(img, [1, shape, shape, 3])/255


def create_model(shape):   
    
    from keras.layers import AveragePooling2D, Dense, Flatten, Dropout
    # from keras import models
    from keras.optimizers import SGD
    # from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras import Model
    
    # from tensorflow.keras.initializers import glorot_uniform
    from tensorflow.keras.regularizers import l2
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import Xception
    model = Xception(input_shape = (shape, shape, 3), include_top = False, weights = 'imagenet')

    x = model.output
    x = AveragePooling2D(pool_size = (2, 2))(x)
    x = Dense(32, activation = 'relu')(x)
    x = Flatten()(x)
    # spostato dopo Flatten
    x = Dropout(0.1)(x)
    # a caso
    x = Dense(128)(x)
    x = Dense(4, activation = 'softmax', kernel_regularizer = l2(.0005))(x)
    model = Model(inputs = model.inputs, outputs = x)
    opt = SGD(lr = 0.0001, momentum = .9)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    return model