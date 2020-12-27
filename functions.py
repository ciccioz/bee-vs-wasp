
# function to convert uploaded images to the format for our model to consume.
# We specifically use the Image class from PIL because the uploaded image is in BytesIO format.
def load_img(input_image, shape):
    img = Image.open(input_image).convert('RGB')
    img = img.resize((shape, shape))
    img = image.img_to_array(img)
    return np.reshape(img, [1, shape, shape, 3])/255

