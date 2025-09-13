import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Carica MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., None].astype(np.float32) / 255.0
x_test  = x_test[..., None].astype(np.float32) / 255.0
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat  = tf.keras.utils.to_categorical(y_test, 10)

# 2️⃣ Crea un modello semplice
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

# 3️⃣ Allena velocemente il modello
model.fit(x_train, y_train_cat, epochs=1, batch_size=128, verbose=2)

# 4️⃣ Valutazione su test set
loss, acc = model.evaluate(x_test, y_test_cat, verbose=2)
print(f"Accuracy su test set: {acc*100:.2f}%")

# 5️⃣ Definizione attacco FGSM
def fgsm_attack(image, label, epsilon=0.1):
    image = tf.convert_to_tensor(image[None], dtype=tf.float32)
    label = tf.convert_to_tensor(label[None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.categorical_crossentropy(label, prediction, from_logits=True)
    gradient = tape.gradient(loss, image)
    adv_image = image + epsilon * tf.sign(gradient)
    adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image[0].numpy()

# 6️⃣ Test attacco su prima immagine
adv_img = fgsm_attack(x_test[0], y_test_cat[0], epsilon=0.1)
plt.subplot(1,2,1)
plt.title(f"Originale: {y_test[0]}")
plt.imshow(x_test[0].squeeze(), cmap='gray')
plt.subplot(1,2,2)
plt.title("Adversarial")
plt.imshow(adv_img.squeeze(), cmap='gray')
plt.show()