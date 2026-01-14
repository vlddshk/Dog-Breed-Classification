import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

class DogBreedClassifier:
    def __init__(self, model_path='models/baseline_model.h5', breed_names_path='breed_names.txt'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(breed_names_path):
            raise FileNotFoundError(f"Breed names file not found: {breed_names_path}")
            
        self.model = load_model(model_path)
        self.input_shape = self.model.layers[0].input_shape[1:]
        
        with open(breed_names_path, 'r') as f:
            self.class_names = f.read().splitlines()

    def predict(self, x, top_k=5, verbose=True):
        if isinstance(x, np.ndarray):
            if x.shape == self.input_shape:
                x = tf.reshape(x, [1] + list(self.input_shape))
        
        pred = self.model.predict(x)
        top_k_pred, top_k_indices = tf.math.top_k(pred, k=top_k)

        predictions = dict()
        for ct in range(top_k):
            name = self.class_names[top_k_indices[0][ct]]
            value = top_k_pred.numpy()[0][ct]
            predictions[name] = value
            if verbose:
                print(f"{name} : {value*100:.2f}%")
        return predictions

    def run_classifier(self, file_paths):
        img_list = []
        valid_paths = []
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    loaded_img = load_img(file_path, target_size=self.input_shape)
                    img_array = img_to_array(loaded_img) / 255.
                    img_list.append(img_array)
                    valid_paths.append(file_path)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
            else:
                print(f"Image not found: {file_path}")

        if not img_list:
            print("No valid images to process.")
            return

        print("")
        result = []
        for idx, val in enumerate(valid_paths):
            print(f"Image file path: {val}")
            result.append(self.predict(img_list[idx], top_k=3))
            print("")

        num_images = len(img_list)
        num_rows = 2 if num_images > 1 else 1
        num_cols = (num_images + 1) // 2

        plt.figure(figsize=(12, 6))
        for idx, val in enumerate(valid_paths):
            plt.subplot(num_rows, num_cols, idx+1)
            img = Image.open(val)
            img.thumbnail((300, 300), Image.LANCZOS)
            plt.imshow(img)
            title_str = "\n".join([f'{k} - {100*v:.2f}%' for k, v in result[idx].items()])
            plt.title(title_str)
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout(pad=2.0)
        plt.show()

def main():
    # Example usage
    file_paths = ['./images/Dog.jpg']
    
    try:
        classifier = DogBreedClassifier()
        classifier.run_classifier(file_paths)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()