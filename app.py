import tensorflow as tf
import os, io, shutil, base64
import numpy as np
from PIL import Image
import streamlit as st

# Model Function
def create_encoder():
    """Create encoder model from VGG19 model
    until "block4_conv1" layer"""

    vgg19 = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )

    vgg19.trainable = False

    encoder = tf.keras.Model(vgg19.input, vgg19.get_layer("block4_conv1").output)
    inputs = layers.Input([224, 224, 3], name="image")
    encoder_out = encoder(inputs)

    return tf.keras.Model(inputs, encoder_out, name="encoder")


def create_decoder():
    """Create decoder model based on inverted encoder structure"""

    config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}
    decoder = tf.keras.Sequential(
        [
            layers.InputLayer((None, None, 512)),
            layers.Conv2D(512, **config),
            layers.UpSampling2D(),
            layers.Conv2D(256, **config),
            layers.Conv2D(256, **config),
            layers.Conv2D(256, **config),
            layers.Conv2D(256, **config),
            layers.UpSampling2D(),
            layers.Conv2D(128, **config),
            layers.Conv2D(128, **config),
            layers.UpSampling2D(),
            layers.Conv2D(64, **config),
            layers.Conv2D(
                3, kernel_size=3, strides=1, padding="same", activation="sigmoid"
            ),
        ]
    )

    return decoder


def create_loss_model():
    """Create loss model from VGG19 similar to encoder model
    to evalute mean & deviation each layer"""

    vgg19 = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )

    vgg19.trainable = False

    layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]

    outputs = [vgg19.get_layer(name).output for name in layer_names]
    loss_model = tf.keras.Model(vgg19.input, outputs)

    inputs = layers.Input([224, 224, 3], name="image")
    loss_model_out = loss_model(inputs)

    return tf.keras.Model(inputs, loss_model_out, name="loss_net")


def mean_std(x, epsilon=1e-5):
    """Calculate mean and standard deviation"""

    axes = [1, 2]

    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.sqrt(variance + epsilon)
    return mean, standard_deviation


def adain(style, content):
    """Computes the AdaIn feature map"""

    content_mean, content_std = mean_std(content)
    style_mean, style_std = mean_std(style)
    adain_map = style_std * (content - content_mean) / content_std + style_mean

    return adain_map

# Image preprocessing

def scale_image(image, size):
    "size specifies the minimum height or width of the output"
    h, w, _ = image.shape
    if h > w:
        image = np.array(Image.fromarray(image).resize((int(h*size//w),
                                                        int(size)),
                                                        Image.Resampling.BICUBIC))
    else:
        image = np.array(Image.fromarray(image).resize((int(size),
                                                        int(w*size//h)),
                                                        Image.Resampling.BICUBIC))
    return image


def central_crop(image):
    '''Apply central crop to image'''
    h, w, _ = image.shape
    minsize = min(h, w)
    h_pad, w_pad = (h - minsize) // 2, (w - minsize) // 2
    image = image[h_pad:h_pad+minsize, w_pad:w_pad+minsize]
    return image

def prepare_image(file_dir, size):
    '''Prepare image to be feed to AdaIn model'''
    image = Image.open(file_dir)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = np.asarray(image)
    image = central_crop(image)
    image = scale_image(image,size)
    image = image /255
    image = np.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)
    return image

# Get the current working directory
current_directory = os.getcwd()

# Define saved model folder path 
saved_model_dir = os.path.join(current_directory, 'Adain 500 Epochs')

model = tf.saved_model.load(saved_model_dir)


# Define content and style directory name
contents_directory = "contents"
styles_directory = "styles"

# Define content_image and style_image variables at the beginning
content_image = None
style_image = None

# Fucntion to upload image
def image_uploader(directory, label):
    uploaded_file = st.file_uploader("", key=f"{label}_uploader", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=f'{label} Image', use_column_width="auto")

        with open(os.path.join(directory, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Return the path of the uploaded image
        return os.path.join(directory, uploaded_file.name)

    # Return None if no image is uploaded
    return None

# Function to clear a specific directory
def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Clear all directories on app start
clear_directory(contents_directory)
clear_directory(styles_directory)

# Create contents and styles directory
create_directory(contents_directory)
create_directory(styles_directory)

st.markdown('<h1 style="line-height:2; color:#007acc; max-width:1900px;">Neural Style Transfer with AdaIN Method</h1>', unsafe_allow_html=True)



col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h4 style= padding-left:50px;"><strong>Upload content image<strong></h4>', unsafe_allow_html=True)
    content_image = image_uploader(contents_directory, 'content')
             
with col2:
    st.markdown('<h4 style= padding-left:50px;"><strong>Upload style image<strong></h4>', unsafe_allow_html=True)
    style_image = image_uploader(styles_directory, 'style')


st.write('')
st.markdown('---')

def reconstruct_image(content_image, style_image):
    # Preparing content and style format
    prepared_content = prepare_image(content_image, 224)
    prepared_style = prepare_image(style_image, 224)
    # Process array to model
    style_encoded = model.encoder(prepared_style)
    content_encoded = model.encoder(prepared_content)
    mapped = adain(style=style_encoded, content=content_encoded)
    reconstructed_image = model.decoder(mapped)
    reconstructed_image = reconstructed_image * 255.0

    image_pil = Image.fromarray(np.uint8(reconstructed_image[0]))
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str

col1, col2,col3 = st.columns([1.6,1,1])

with col1:
    pass
with col2:
    if st.button('Execute!'):
        if content_image is not None and style_image is not None:
            img_str = reconstruct_image(content_image,style_image)
with col3:
    pass


if 'img_str' in locals():
    st.image(
        f"data:image/jpeg;base64,{img_str}",
        caption='Reconstructed Image Resized in 224 x 224',
        use_column_width='True')