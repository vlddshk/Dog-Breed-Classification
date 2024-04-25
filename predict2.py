import pathlib
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model('baseline_model.h5')
input_shape = model.layers[0].input_shape[1:]

# Define the predict function
#This Python function, predict, is used to predict the top k class labels for an input x using a trained machine learning model.
def predict(x, top_k=5, verbose=True):
    if isinstance(x, np.ndarray):
        assert x.shape == input_shape
        x = tf.reshape(x, [1] + list(input_shape))

    # Predict
    pred = model.predict(x)
    top_k_pred, top_k_indices = tf.math.top_k(pred, k=top_k)

    # Display the prediction
    predictions = dict()
    for ct in range(top_k):
        with open('breed_names.txt', 'r') as f:
          class_names = f.read().splitlines()

        name = class_names[top_k_indices[0][ct]]
        value = top_k_pred.numpy()[0][ct]
        predictions[name] = value
        if verbose:
            print(f"{name} : {value*100:.2f}%")
    return predictions

# Define the run_classifier function
#This code defines a function called run_classifier that takes in a list of file paths. 
#It loads the images from these file paths, resizes them, and predicts the breed of each image using a pre-trained model. 
#The predicted breeds are then displayed along with their corresponding probabilities. 
#The images are displayed in a grid with their corresponding predictions.
def run_classifier(file_paths):
    # Load the images
    img_list = [img_to_array(load_img(file_path, target_size=input_shape)) / 255. for file_path in file_paths]

    # Predict
    print("")
    result = []
    for idx, val in enumerate(file_paths):
        print(f"Image file path: {val}")
        result.append(predict(img_list[idx], top_k=3))
        print("")

    num_images = len(img_list)
    num_rows = 2 if num_images > 1 else 1
    num_cols = (num_images + 1) // 2

    for idx, val in enumerate(file_paths):
        plt.subplot(num_rows, num_cols, idx+1)
        img = Image.open(val)
        img.thumbnail((120, 120), Image.LANCZOS)# resizes image in-place
        plt.imshow(img)
        title_str = "\n".join([f'{k} - {100*v:.2f}%' for k, v in result[idx].items()])
        plt.title(title_str)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout(pad=2.0)
    plt.show()

def main(file_paths):
    run_classifier(file_paths)


if __name__ == "__main__":
    # Specify the file paths directly in the code
    file_paths = ['Dog.jpg']
    main(file_paths)