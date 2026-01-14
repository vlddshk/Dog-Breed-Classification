# Dog Breed Classification

This project uses Deep Learning to identify dog breeds from images. The model is built on the **InceptionV3** architecture and trained on the **Stanford Dogs Dataset**.

## Features

*   **Model Training (`train.py`)**: Script for training the model on a custom or downloaded dataset using TensorFlow/Keras.
*   **Prediction (`inference.py`)**: CLI tool for predicting dog breeds on local images.
*   **Telegram Bot (`bot.py`)**: Interactive bot that accepts photos from users and returns the predicted breed.

## Project Structure

```
.
├── train.py               # Model training script
├── inference.py           # Prediction script (CLI)
├── bot.py                 # Telegram bot
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (Bot Token)
├── breed_names.txt        # List of breed names
├── models/                # Saved model weights
├── dataset/               # Dataset folder (for training)
└── README.md              # Documentation
```

## Installation

1.  **Clone the repository** (if using git):
    ```bash
    git clone https://github.com/vlddshk/Dog-Breed-Classification
    cd Dog-Breed-Classification
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Telegram Bot Setup**:
    *   Create a `.env` file in the project root.
    *   Add your token obtained from @BotFather:
        ```env
        TGBOTTOKEN=your_telegram_bot_token_here
        ```

## Usage

### 1. Training the Model
If you want to train the model yourself:
    Run the script:
```bash
python train.py
```
After training completes, the best model will be saved in `models/baseline_model.h5`.

### 2. Photo Prediction (CLI)
To quickly test the model on a local photo:
1.   In `inference.py` (in the `main` function), you can specify the image path or pass a list of files.
2.    Run:
```bash
python inference.py
```

### 3. Running the Telegram Bot
To activate the bot:
1.  Ensure the `.env` file is configured.
2.  Run the bot:
    ```bash
    python bot.py
    ```
3.  Open Telegram and send the bot a photo of a dog

## About the Dataset
*   **Number of categories**: 120 breeds
*   **Number of images**: 20,580
*   **Source**: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
