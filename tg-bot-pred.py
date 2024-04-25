import os
import numpy as np
import telebot
from PIL import Image
from skimage.transform import resize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the model(завантажуємо модель)
model = load_model('baseline_model.h5')
input_shape = model.layers[0].input_shape[1:]

# Load breed names(завантажуємо назви пород)
with open('breed_names.txt', 'r') as f:
    breed_names = f.read().splitlines()

# Initialize the Telegram Bot(додаємо токен телеграм бота)
bot = telebot.TeleBot("TGBOTTOKEN")

# Handle '/start' and '/help'(додаймо відповідь на старт)
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привіт, надішли мені фото свого улюблинця а я спробую відгадати його породу")

# Handle photo messages (обробляємо надіслані боту фото)
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        # Download photo
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        photo_path = 'temp_photo.jpg'
        with open(photo_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        # Load and preprocess the image
        img = Image.open(photo_path)
        img = img_to_array(img)
        img = resize(img, input_shape)
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img)[0]
        top_indices = pred.argsort()[-3:][::-1]

        # Send prediction
        response = "Ось 3 найбільш схожих порід:\n \n сподіваюсь я правильно визначив та вам все сподобалось:)"
        for i in top_indices:
            response += f"{breed_names[i]} - {pred[i] * 100:.2f}%\n"

        bot.reply_to(message, response)

    except Exception as e:
        print(e)
        bot.reply_to(message, "Вибачте, щось пішло не за планом( спробуйте ще раз.")

# Handle all other messages(додаємо відповідь на всі інші повідомлення)
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, "Нажаль поки що я можу відгадувати лише породи собак.")

# Polling(запускаємо бот)
bot.polling()
