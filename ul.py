import tensorflow as tf
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
import numpy as np

# Load CNN model
cnn_model = load_model("bestmodel.h5")  # Adjust the path accordingly

# Image preprocessing function
def preprocess_ultrasound_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))  # Resize the image
    img_arr = tf.keras.utils.img_to_array(img)  # Convert image to array
    
    # Debugging: Check the shape of the image after loading
    print(f"Image shape after loading: {img_arr.shape}")
    
    img_arr = preprocess_input(img_arr)  # Apply necessary preprocessing (e.g., MobileNetV2 preprocessing)
    
    # Debugging: Check the shape of the image after preprocessing
    print(f"Image shape after preprocessing: {img_arr.shape}")
    
    input_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension for the model
    ultrasound_pred = cnn_model.predict(input_arr)[0][0]  # Predict the probability
    
    print(f"Ultrasound Model Raw Prediction: {ultrasound_pred}")
    
    return ultrasound_pred
