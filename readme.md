Struttura del progetto

Di seguito trovi la descrizione in Markdown dei file principali del progetto e del loro scopo.

model.py

Contiene solo la definizione del modello CNN.

Funzione principale: create_mnist_cnn()

Restituisce un modello compilato, pronto all'uso (loss, optimizer e metriche configurate).

Motivo: separare la struttura del modello dal resto del codice permette di sostituirlo facilmente (es. provare un modello più grande o diverso) senza toccare l’addestramento o gli attacchi.

# Esempio di interfaccia (non il codice completo)
def create_mnist_cnn() -> tf.keras.Model:
    """
    Ritorna un modello tf.keras compilato per MNIST.
    """
    ...

data.py

Gestisce il caricamento e il preprocessing dei dati.

Funzione principale: load_mnist_data()

Restituisce:

set di training (immagini e etichette in one-hot)

set di test (immagini e etichette in one-hot)

le etichette originali del test set (interi, utile per visualizzare immagini)

Motivo: se un giorno vuoi cambiare dataset (es. FashionMNIST), basta modificare questo file senza toccare modello o attacco.

# Esempio di interfaccia
def load_mnist_data():
    """
    Ritorna: (x_train, y_train_onehot), (x_test, y_test_onehot), y_test_orig
    - x_* in formato float32 normalizzato
    - y_* one-hot encoded (per training e test)
    - y_test_orig come array di label intere per visualizzazione
    """
    ...

attack.py

Contiene la funzione fgsm_attack() che genera immagini adversariali.

Parametri: modello, immagine, etichetta, epsilon.

Implementazione: usa tf.GradientTape per calcolare il gradiente della loss rispetto all’immagine, poi crea la perturbazione FGSM (sign(gradient) * epsilon) e ritorna l’immagine perturbata (clippata nello stesso intervallo di input).

Motivo: avere l’attacco separato permette di testare diversi epsilon o attacchi senza modificare il modello o il dataset.

# Esempio di interfaccia
def fgsm_attack(model: tf.keras.Model, image: np.ndarray, label: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Restituisce l'immagine adversariale generata con FGSM.
    - image: singola immagine o batch
    - label: label corrispondente (one-hot o integer, a seconda dell'implementazione)
    - epsilon: intensità della perturbazione
    """
    ...

main.py

Coordinatore di tutto: script eseguibile che mette insieme le parti.

Flusso tipico:

Carica i dati tramite data.load_mnist_data()

Crea il modello con model.create_mnist_cnn()

Addestra il modello (model.fit(...))

Valuta l’accuracy sul test set (model.evaluate(...))

Genera e mostra un esempio di attacco FGSM (usa attack.fgsm_attack(...), visualizza immagine originale, perturbazione e immagine adversariale; mostra predizioni)

(Opzionale) salva il modello in .h5

Motivo: separare la logica dall’implementazione dei dettagli rende il codice più leggibile e manutenibile.

# Esempio di skeleton
if _name_ == "_main_":
    # 1. dati
    (x_train, y_train), (x_test, y_test), y_test_orig = load_mnist_data()

    # 2. modello
    model = create_mnist_cnn()

    # 3. addestramento
    model.fit(x_train, y_train, epochs=..., validation_split=...)

    # 4. valutazione
    model.evaluate(x_test, y_test)

    # 5. attacco FGSM: esempio su una immagine di test
    adv = fgsm_attack(model, x_test[0:1], y_test[0:1], epsilon=0.1)
    # visualizza / confronta predizioni

    # 6. salva (opzionale)
    model.save("mnist_cnn.h5")

Note aggiuntive e best practice

Mantieni ogni file piccolo e con una responsabilità unica (single responsibility principle).

Documenta le funzioni con docstring in italiano o inglese (coerenza nel progetto).

Per esperimenti ripetibili, fissa il seed di NumPy/TensorFlow nello script main.py.

Considera di aggiungere un file utils.py se emergono funzioni di utilità condivise (es. funzioni di visualizzazione).

Versiona il progetto (es. requirements.txt o pyproject.toml) per rendere riproducibile l’ambiente.


```python
def fgsm_attack(model, image, label, epsilon=0.1):
    

    # 1) Aggiungo la dimensione batch: shape (H,W,C) -> (1,H,W,C)
    image = tf.convert_to_tensor(image[None], dtype=tf.float32)
    # Perché: TF calcola gradienti su tensori; la maggior parte dei modelli
    # si aspetta un batch. Qui lavoriamo su un singolo esempio ma lo
    # trasformiamo in batch size 1.

    # 2) Preparo la label in formato tensore (shape (1, n_classes) se one-hot)
    label = tf.convert_to_tensor(label[None], dtype=tf.float32)
    # Nota: qui si assume che `label` sia one-hot. Se usi etichette sparse
    # (es. intero), serve usare loss diversa (SparseCategorical...).

    # 3) Attivo il tracciamento dei gradienti rispetto all'immagine
    with tf.GradientTape() as tape:
        tape.watch(image)
        # tape.watch dice a TF di tracciare i gradienti rispetto a `image`.
        # Senza questo non potrei ottenere grad(loss, image).

        # 4) Forward pass: predizione del modello su batch di dimensione 1
        prediction = model(image)
        # `prediction` ha shape (1, n_classes). Può essere logits (no softmax)
        # o probabilità (softmax) a seconda della definizione del modello.

        # 5) Calcolo la loss fra label e predizione
        loss = tf.keras.losses.categorical_crossentropy(label, prediction, from_logits=True)
        # from_logits=True indica che `prediction` sono logits non normalizzati.
        # Se il tuo modello termina con softmax, usa from_logits=False.
        # Il risultato `loss` ha shape (1,), loss scalare per l'esempio.

    # 6) Calcolo del gradiente della loss rispetto all'immagine
    gradient = tape.gradient(loss, image)
    # `gradient` ha la stessa shape di `image`, cioè (1,H,W,C).
    # Rappresenta la derivata parziale della loss rispetto a ciascun pixel.

    # 7) Costruisco la perturbazione: segno del gradiente
    adv_image = image + epsilon * tf.sign(gradient)
    # tf.sign(gradient) restituisce -1, 0 o +1 per ciascun elemento.
    # Moltiplicando per epsilon ottieni la perturbazione FGSM element-wise:
    # se il gradiente è positivo, aggiungi epsilon; se negativo, sottrai epsilon.
    # Questo è FGSM (single-step, L_infty bounded).

    # 8) Clamp dei valori per restare nell'intervallo valido
    adv_image = tf.clip_by_value(adv_image, 0, 1)
    # Se le immagini sono normalizzate in [0,1], qui si evita overflow o valori
    # fuori range che il modello non si aspetta.

    # 9) Ritorno l'immagine adversariale come array NumPy senza la dimensione batch
    return adv_image[0].numpy()
```
