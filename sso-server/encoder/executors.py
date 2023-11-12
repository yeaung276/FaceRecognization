import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor

import env

encoder = None

def load_model():
    global encoder
    encoder = tf.keras.models.load_model(env.MODEL_PATH)

executor = ProcessPoolExecutor(max_workers=1, initializer=load_model)