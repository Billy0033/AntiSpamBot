import logging
import asyncio
from aiogram import Bot, Dispatcher, Router, types
from aiogram.types import Message
from aiogram.filters import Command
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Инициализация логгера
logging.basicConfig(level=logging.INFO)

# Токен вашего Telegram-бота
BOT_TOKEN = "7547438624:AAEUTUI2L2bDm1Fv6cG9uNXwCcdEwXvRTk0"

# Параметры модели
MODEL_PATH = "spam_classifier.h5"
TOKENIZER_PATH = "tokenizer.json"
MAX_SEQUENCE_LENGTH = 100  # Укажите максимальную длину последовательности

# Загрузка модели и токенизатора
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
router = Router()  # Создаем роутер

def predict_spam_probability(text: str) -> float:
    """Функция для предсказания вероятности спама."""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    prediction = model.predict(padded_sequence)
    return float(prediction[0][0])

@router.message(Command("start"))
async def handle_start(message: Message):
    """Обработчик команды /start, приветственное сообщение."""
    await message.reply("Привет! Я антиспам-бот. Пришли мне сообщение, и я определю, является ли оно спамом.")

@router.message()
async def handle_message(message: Message):
    """Обработчик входящих сообщений."""
    # Пропускаем команды типа /start
    if message.text.startswith("/"):
        return

    # Проверяем, переслано ли сообщение
    if message.forward_from or message.forward_from_chat:
        text = message.text
        if not text:
            await message.reply("Пересланное сообщение не содержит текста для анализа.")
            return
        
        spam_probability = predict_spam_probability(text)
        response = f"Вероятность спама в пересланном сообщении: {spam_probability:.2%}"
        await message.reply(response)
    else:
        # Обычное сообщение
        text = message.text
        spam_probability = predict_spam_probability(text)
        response = f"Вероятность спама: {spam_probability:.2%}"
        await message.reply(response)

async def main():
    # Настройка диспетчера и подключение роутера
    dp.include_router(router)
    
    logging.info("Бот запущен")
    await bot.delete_webhook(drop_pending_updates=True)  # Удаление вебхука, если был установлен
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())




# import logging
# import asyncio
# from aiogram import Bot, Dispatcher, Router, types
# from aiogram.types import Message
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import json

# # Инициализация логгера
# logging.basicConfig(level=logging.INFO)

# # Токен вашего Telegram-бота
# BOT_TOKEN = "7547438624:AAEUTUI2L2bDm1Fv6cG9uNXwCcdEwXvRTk0"

# # Параметры модели
# MODEL_PATH = "spam_classifier.h5"
# TOKENIZER_PATH = "tokenizer.json"
# MAX_SEQUENCE_LENGTH = 100  # Укажите максимальную длину последовательности

# # Загрузка модели и токенизатора
# model = load_model(MODEL_PATH)
# with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
#     tokenizer_data = json.load(f)
#     tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)

# # Инициализация бота и диспетчера
# bot = Bot(token=BOT_TOKEN)
# dp = Dispatcher()
# router = Router()  # Создаем роутер

# def predict_spam_probability(text: str) -> float:
#     """Функция для предсказания вероятности спама."""
#     sequence = tokenizer.texts_to_sequences([text])
#     padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
#     prediction = model.predict(padded_sequence)
#     return float(prediction[0][0])

# @router.message()
# async def handle_message(message: Message):
#     """Обработчик входящих сообщений."""
#     text = message.text
#     spam_probability = predict_spam_probability(text)
#     response = f"Вероятность спама: {spam_probability:.2%}"
#     await message.reply(response)

# async def main():
#     # Настройка диспетчера и подключение роутера
#     dp.include_router(router)
    
#     logging.info("Бот запущен")
#     await bot.delete_webhook(drop_pending_updates=True)  # Удаление вебхука, если был установлен
#     await dp.start_polling(bot)

# if __name__ == "__main__":
#     asyncio.run(main())
