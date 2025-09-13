import tensorflow as tf
import os

def create_mnist_cnn():
    """Crea un modello CNN semplice per MNIST"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)  # logits
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    if not os.path.exists("mnist_cnn.h5"):
      model.save("mnist_cnn.h5")
      print("Modello salvato come mnist_cnn.h5")
    return model