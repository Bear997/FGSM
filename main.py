import matplotlib.pyplot as plt
from model import create_mnist_cnn
from data import load_mnist_data
from attack import fgsm_attack
import tensorflow as tf

# 1️⃣ Carica dati
(x_train, y_train), (x_test, y_test_cat), y_test_raw = load_mnist_data()

# 2️⃣ Crea modello
model = create_mnist_cnn()

# 3️⃣ Addestra il modello velocemente
model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=2)

# 4️⃣ Valuta accuracy
loss, acc = model.evaluate(x_test, y_test_cat, verbose=2)
print(f"Accuracy su test set: {acc*100:.2f}%")

epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

for eps in epsilons:
    adv_img = fgsm_attack(model, x_test[0], y_test_cat[0], epsilon=eps)
    
    # Predizione originale
    orig_pred = model(x_test[0][None], training=False)
    orig_class = tf.argmax(orig_pred, axis=1).numpy()[0]
    
    # Predizione adversariale
    adv_pred = model(adv_img[None], training=False)
    adv_class = tf.argmax(adv_pred, axis=1).numpy()[0]
    
    print(f"Epsilon: {eps:.2f} | Classe originale: {orig_class} | Classe adversariale: {adv_class}")
    
    # Visualizza immagini
    plt.subplot(1,2,1)
    plt.title(f"Originale: {orig_class}")
    plt.imshow(x_test[0].squeeze(), cmap='gray')
    
    plt.subplot(1,2,2)
    plt.title(f"Adv (eps={eps:.2f}): {adv_class}")
    plt.imshow(adv_img.squeeze(), cmap='gray')
    
    plt.show()