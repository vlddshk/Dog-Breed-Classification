import pathlib
import imageio
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

# Configuration
TRAIN_DATA_DIR = pathlib.Path("dataset")
IMG_WIDTH, IMG_HEIGHT = 224, 224
CHANNELS = 3
BATCH_SIZE = 64
NUM_IMAGES_PREVIEW = 50

def get_images(image_dir):
    image_index = 0
    images_arr = []
    labels = []
    
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} not found.")
        return np.array([]), np.array([])
        
    subdirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    
    for type_name in subdirs[:50]: # Limit to 50 types for preview
        path = os.path.join(image_dir, type_name)
        type_images = os.listdir(path)
        
        label = type_name.split('-')[1] if '-' in type_name else type_name
        
        for image_name in type_images[:1]:
            image_file = os.path.join(path, image_name)
            try:
                image_data = imageio.imread(image_file)
                image_resized = resize(image_data, (IMG_WIDTH, IMG_HEIGHT), anti_aliasing=True)
                images_arr.append(image_resized.flatten())
                labels.append(label)
                print(f"{type_name} : {image_name}")
            except Exception as e:
                print(f"Error reading {image_file}: {e}")

    return np.array(images_arr), np.array(labels)

def plot_images(instances, images_per_row=10, **options):
    if len(instances) == 0:
        return
        
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(IMG_WIDTH, IMG_HEIGHT, CHANNELS) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    
    row_images = []
    
    n_empty = n_rows * images_per_row - len(instances)
    for _ in range(n_empty):
        images.append(np.zeros((IMG_WIDTH, IMG_HEIGHT, CHANNELS)))

    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    
    image = np.concatenate(row_images, axis=0)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(image, **options)
    plt.axis("off")
    os.makedirs('./images', exist_ok=True)
    plt.savefig('./images/dogs_images.png', transparent=True, bbox_inches='tight', dpi=900)
    # plt.show() # Commented out to avoid blocking execution in non-interactive environments

def train():
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"Dataset directory '{TRAIN_DATA_DIR}' not found. Please ensure the dataset is present.")
        return

    images, labels = get_images(TRAIN_DATA_DIR)
    if len(images) > 0:
        plot_images(images)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        validation_split=0.2,
    )
    
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=1337
    )

    valid_generator = valid_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=1337
    )

    nb_train_samples = train_generator.samples
    nb_valid_samples = valid_generator.samples
    
    base_model = applications.InceptionV3(
        include_top=False, 
        input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS), 
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(120, activation='softmax'))
    
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    checkpoint = ModelCheckpoint(
        './models/baseline_model.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto',
        save_weights_only=False,
        save_freq='epoch'
    )

    earlystop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=3,
        verbose=1,
        mode='auto'
    )

    csvlogger = CSVLogger(
        filename="./logs/training_csv.log",
        separator=",",
        append=False
    )

    reduceLR = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1,
        mode='auto'
    )

    callbacks = [checkpoint, earlystop, csvlogger, reduceLR]

    history = model.fit(
        train_generator,
        epochs=30,
        steps_per_epoch=nb_train_samples // BATCH_SIZE,
        validation_data=valid_generator,
        validation_steps=nb_valid_samples // BATCH_SIZE,
        verbose=2,
        callbacks=callbacks,
        shuffle=True
    )

    (eval_loss, eval_accuracy) = model.evaluate(valid_generator, batch_size=BATCH_SIZE, verbose=1)
    print('Validation Loss: ', eval_loss)
    print('Validation Accuracy: ', eval_accuracy)
    plt.figure()
    plt.title('Model Accuracy')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.savefig('./images/baseline_acc_epoch.png', transparent=False, bbox_inches='tight', dpi=900)
    
    plt.figure()
    plt.title('Model Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig('./images/baseline_loss_epoch.png', transparent=False, bbox_inches='tight', dpi=900)

if __name__ == "__main__":
    train()
