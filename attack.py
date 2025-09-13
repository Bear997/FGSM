import tensorflow as tf

def fgsm_attack(model, image, label, epsilon=0.1):
    """Genera un'immagine adversariale con FGSM"""
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