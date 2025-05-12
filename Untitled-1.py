# %%
import numpy as np
import matplotlib.pyplot as plt
import os 
import math
import shutil
import zipfile
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf

# Set root directory
ROOT_DIR = r"F:\ULTRASOUND\PCOS"

# %% Count number of images per class
def get_image_counts():
    number_of_images = {}
    for dir in os.listdir(ROOT_DIR):
        number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))
        print(dir, ":", number_of_images[dir])
    return number_of_images

number_of_images = get_image_counts()

# %% Create data folder

def datafolder(path, split):
    if not os.path.exists("./" + path):
        os.mkdir("./" + path)
        for dir in os.listdir(ROOT_DIR):
            os.makedirs("./" + path + "/" + dir)
            count = max(1, math.floor(split * number_of_images[dir]) - 1)
            imgs = np.random.choice(os.listdir(os.path.join(ROOT_DIR, dir)), size=count, replace=False)
            for img in imgs:
                src = os.path.join(ROOT_DIR, dir, img)
                dst = os.path.join("./" + path, dir)
                shutil.copy(src, dst)
    else:
        print(f"Folder '{path}' already exists.")

# %% Create train, val, test folders
datafolder("train", 0.7)
datafolder("val", 0.15)
datafolder("test", 0.15)

# %% Preprocessing functions
def preprocessingImage1(path):
    image_data = ImageDataGenerator(
        zoom_range=0.2,
        shear_range=0.2,
        preprocessing_function=preprocess_input,
        horizontal_flip=True
    )
    return image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='binary')

def preprocessingImage2(path):
    image_data = ImageDataGenerator(preprocessing_function=preprocess_input)
    return image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)

# %% Load data
train_data = preprocessingImage1("./train")
val_data = preprocessingImage2("./val")
test_data = preprocessingImage2("./test")

# %% Build CNN model with MobileNet
base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# %% Callbacks
callbacks = [
    ModelCheckpoint("bestmodel.h5", save_best_only=True, monitor="val_accuracy", mode='max'),
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
]

# %% Train model
hist = model.fit(
    train_data,
    epochs=30,
    validation_data=val_data,
    callbacks=callbacks
)

# %% Load and evaluate best model
model = load_model("bestmodel.h5")
loss, acc = model.evaluate(test_data)
print(f"Model Test Accuracy: {acc * 100:.2f}%")

# %% Prediction function
def predictimage(path):
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = preprocess_input(img_array)
    input_arr = np.expand_dims(img_array, axis=0)

    pred = model.predict(input_arr)[0][0]
    if pred > 0.5:
        print(f"Prediction: Not Affected ({pred:.2f})")
    else:
        print(f"Prediction: Affected ({pred:.2f})")


    plt.imshow(img)
    plt.title("Input Image")
    plt.axis("off")
    plt.show()

# %% Predict sample image
predictimage(r"F:\ULTRASOUND\img_0_84.jpg")


predictimage(r"F:\ULTRASOUND\arch.jpg")
# %%
