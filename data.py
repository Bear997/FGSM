import tensorflow as tf
import numpy as np

def load_mnist_data():
    """Carica MNIST e normalizza i valori tra 0 e 1"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., None].astype(np.float32) / 255.0
    x_test  = x_test[..., None].astype(np.float32) / 255.0
    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_test_cat  = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train_cat), (x_test, y_test_cat), y_test