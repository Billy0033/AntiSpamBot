import tensorflow as tf

# Загрузка модели из .h5
model = tf.keras.models.load_model("spam_classifier.h5")

# Сохранение модели в формате SavedModel
model.export("saved_model")  # "saved_model" - это директория, где будет сохранена модель
