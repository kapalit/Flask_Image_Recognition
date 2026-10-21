"""Loading of the trained Keras digit model plus image preprocessing and prediction."""

import os

import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_model.h5")

# Loaded once at import time so every request reuses the same model.
model = load_model(MODEL_PATH)


def preprocess_img(img_path):
    """Open an image (path or file-like object) and convert it to a model input batch.

    The image is converted to RGB, resized to 224x224, scaled to the 0-1 range and
    reshaped to a batch of one: shape (1, 224, 224, 3).
    """
    op_img = Image.open(img_path).convert("RGB")
    img_resize = op_img.resize(IMG_SIZE)
    img2arr = img_to_array(img_resize) / 255.0
    return img2arr.reshape(1, *IMG_SIZE, 3)


def predict_result(predict):
    """Return the predicted digit (index of the highest class score) for a batch of one."""
    pred = model.predict(predict, verbose=0)
    return np.argmax(pred[0], axis=-1)
