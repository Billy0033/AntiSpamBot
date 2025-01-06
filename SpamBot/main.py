import tf2onnx
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Параметры
MAX_VOCAB_SIZE = 10000  # Максимальный размер словаря
MAX_SEQUENCE_LENGTH = 100  # Максимальная длина последовательности
BATCH_SIZE = 32  # Размер батча

# Загрузка данных из JSON-файла
def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8-sig') as f:  # Используем utf-8-sig
        data = json.load(f)
    texts = [entry["Text"] for entry in data]
    labels = [1 if entry["Label"] == "spam" else 0 for entry in data]  # 1 = spam, 0 = ham
    return texts, labels

# Обработка данных
def preprocess_data(texts, labels):
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return padded_sequences, np.array(labels), tokenizer

# Построение модели
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(MAX_VOCAB_SIZE, 16, input_length=MAX_SEQUENCE_LENGTH),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Регуляризация (Dropout)
        tf.keras.layers.Dense(1, activation='sigmoid')  # Для бинарной классификации
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Основной скрипт
if __name__ == "__main__":
    # Укажите путь к вашему JSON-файлу
    json_file = "spam.json"  # Замените на свой путь

    # Загрузка и обработка данных
    texts, labels = load_data(json_file)
    sequences, labels, tokenizer = preprocess_data(texts, labels)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    # Создание датасетов с перетасовкой
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

    # Построение и обучение модели
    model = build_model()
    model.summary()

    # Обучение модели
    model.fit(train_dataset, epochs=500, validation_data=test_dataset)

    # Тестирование модели
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Точность на тестовых данных: {accuracy:.2f}")

    # Сохранение токенизатора и модели
    model.save("spam_classifier.h5")
    with open("tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer.to_json(), f, ensure_ascii=False)

    print("Модель и токенизатор сохранены.")


import tensorflow as tf
import tf2onnx

# Параметры
h5_model_path = "spam_classifier.h5"  # Путь к вашей .h5 модели
onnx_model_path = "spam_classifier.onnx"  # Путь для сохранения ONNX модели

# Загрузка модели из файла .h5
model = tf.keras.models.load_model(h5_model_path)

# Определение входной сигнатуры
input_signature = [tf.TensorSpec([None, 100], tf.float32, name="input")]

# Конвертация в ONNX
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path=onnx_model_path
)

print(f"Модель успешно сохранена в формате ONNX: {onnx_model_path}")


