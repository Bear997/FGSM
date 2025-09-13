import matplotlib.pyplot as plt
from model import create_mnist_cnn
from data import load_mnist_data
from attack import fgsm_attack
import tensorflow as tf
import random

# 1️⃣ Carica dati
(x_train, y_train), (x_test, y_test_cat), y_test_raw = load_mnist_data()

# 2️⃣ Crea modello
model = create_mnist_cnn()

# 3️⃣ Addestra il modello velocemente
model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=2)

# 4️⃣ Valuta accuracy
loss, acc = model.evaluate(x_test, y_test_cat, verbose=2)
print(f"Accuracy su test set: {acc*100:.2f}%")

# Lista di epsilon da testare


num_images = 8
epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.9]

# Seleziona casualmente 'num_images' indici dal test set
indices = random.sample(range(x_test.shape[0]), num_images)

for idx in indices:
    orig_img = x_test[idx]
    orig_label = y_test_raw[idx]
    orig_label_cat = y_test_cat[idx]

    plt.figure(figsize=(len(epsilons)*2, 4))
    
    # Prima colonna: immagine originale
    plt.subplot(2, len(epsilons)+1, 1)
    plt.title(f"Orig: {orig_label}")
    plt.imshow(orig_img.squeeze(), cmap='gray')
    plt.axis('off')
    
    # Ciclo sulle epsilon
    for i, eps in enumerate(epsilons):
        adv_img = fgsm_attack(model, orig_img, orig_label_cat, epsilon=eps)
        adv_pred = tf.argmax(model(adv_img[None], training=False), axis=1).numpy()[0]

        plt.subplot(2, len(epsilons)+1, i+2)
        plt.title(f"eps={eps:.2f}\nPred:{adv_pred}")
        plt.imshow(adv_img.squeeze(), cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()