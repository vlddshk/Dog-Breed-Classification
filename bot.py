import os
import uuid
import numpy as np
import telebot
from PIL import Image
from skimage.transform import resize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('TGBOTTOKEN')
if not TOKEN:
    print("Error: TGBOTTOKEN not found in .env file or environment variables.")
    exit(1)

try:
    model = load_model('./models/baseline_model.h5')
    input_shape = model.layers[0].input_shape[1:]
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

try:
    with open('breed_names.txt', 'r') as f:
        breed_names = f.read().splitlines()
except FileNotFoundError:
    print("Error: breed_names.txt not found.")
    exit(1)

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привіт! Надішли мені фото свого улюбленця, а я спробую відгадати його породу.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    temp_filename = ""
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        temp_filename = f"./images/temp_{uuid.uuid4()}.jpg"
        
        os.makedirs(os.path.dirname(temp_filename), exist_ok=True)

        with open(temp_filename, 'wb') as new_file:
            new_file.write(downloaded_file)

        img = Image.open(temp_filename)
        img = img_to_array(img)
        img = resize(img, input_shape)
        img = np.expand_dims(img, axis=0)
        
        pred = model.predict(img)[0]
        top_indices = pred.argsort()[-3:][::-1]

        response = "Ось 3 найбільш схожих породи:\n\nСподіваюсь, я правильно визначив!\n"
        for i in top_indices:
            response += f"{breed_names[i]} - {pred[i] * 100:.2f}%\n"

        bot.reply_to(message, response)

    except Exception as e:
        print(f"Error processing image: {e}")
        bot.reply_to(message, "Вибачте, щось пішло не так. Спробуйте ще раз або надішліть інше фото.")
    
    finally:
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except OSError as e:
                print(f"Error deleting temp file {temp_filename}: {e}")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, "На жаль, поки що я можу відгадувати лише породи собак по фото.")

if __name__ == "__main__":
    print("Bot started...")
    bot.polling()
