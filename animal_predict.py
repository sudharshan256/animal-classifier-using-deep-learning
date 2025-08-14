import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("C:\\Users\\skillrack.AIDSLABII\\Downloads\\archive\\animal_classifier_model.h5")

img_path = "C:\\Users\\skillrack.AIDSLABII\\Downloads\\archive\\Wolf-Spider-2-scaled-1.jpeg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = tf.argmax(predictions[0]).numpy()

class_names = ["cat", "dog", ...]  # same order as training folder names
print("Predicted animal:", class_names[predicted_class])
