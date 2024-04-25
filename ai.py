import pathlib
import imageio
import os
import pandas as pd
import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from IPython.display import SVG
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

#train_data_dir: It is set to the path "images" using pathlib.Path.
train_data_dir = pathlib.Path("images")
img_width, img_height = 224, 224 
#channels: It is set to 3, representing the number of color channels (RGB).
channels = 3
#batch_size: It is set to 64, indicating the number of images processed in each batch during training.
batch_size = 64
#num_images: It is set to 50, specifying the number of images to be processed.
num_images= 50
#image_arr_size: It calculates the size of each image array by multiplying img_width, img_height, and channels.
image_arr_size= img_width * img_height * channels

#This Python function, get_images, takes an image_dir path as input and returns two arrays: images and labels.
def get_images(image_dir):

    image_index = 0
    image_arr_size= img_width * img_height * channels
    images = np.ndarray(shape=(num_images, image_arr_size))
    labels = np.array([])                       

    for type in os.listdir(image_dir)[:50]:
        path = os.path.join(image_dir, type)
        if os.path.isdir(path):
            type_images = os.listdir(path)
            if '-' in type:
                labels= np.append(labels, type.split('-')[1])
            else:
                labels= np.append(labels, type)
            
            for image in type_images[:1]:
                image_file = os.path.join(image_dir, type, image)
                image_data = imageio.imread(image_file)
                image_resized = resize(image_data, (img_width, img_height), anti_aliasing=True)
                images[image_index, :] = image_resized.flatten()
                print (type, ':', image)
                image_index += 1

    return (images, labels)

#This is a function that takes in a list of instances, and plots them as images in a grid.
def plot_images(instances, images_per_row=10, **options):
    size = img_width
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(img_width, img_height, channels) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((img_width, img_height * n_empty)))
    for row in range(n_rows):
        if (row == len(instances)/images_per_row):
            break
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.figure(figsize=(20,20))
    plt.imshow(image, **options)
    plt.axis("off")
    plt.savefig('dogs_images.png', transparent= True, bbox_inches= 'tight', dpi= 900)
    plt.show()

images, labels = get_images(train_data_dir)
plot_images(images)

#his is a code snippet that creates an instance of the ImageDataGenerator class from the Keras library. 
#This class is used for data augmentation on image data.
train_datagen = ImageDataGenerator(
    #This scales pixel values to the range of 0 to 1.
    rescale= 1./255,
    #This applies a random shearing transformation to the images, which can help increase the diversity of the training data.
    shear_range= 0.2,
    #This applies a random zoom transformation to the images.
    zoom_range= 0.2,
    #This randomly flips the images horizontally.
    horizontal_flip= True,
    rotation_range= 20,
    width_shift_range= 0.2,
    height_shift_range= 0.2,   
    validation_split=0.2,

)
#This is a code snippet that creates an instance of the ImageDataGenerator class from the Keras library, similar to the previous snippet.
valid_datagen = ImageDataGenerator(
    rescale= 1./255, 
    #This specifies that 20% of the data will be used for validation.
    validation_split=0.2,
)

#This is a part of code that creates a generator for training data using the flow_from_directory method of the ImageDataGenerator instance train_datagen.
train_generator = train_datagen.flow_from_directory(  
    train_data_dir,  
    target_size= (img_width, img_height), 
    color_mode= 'rgb',
    batch_size= batch_size,  
    #This specifies that the labels should be encoded as one-hot vectors.
    class_mode= 'categorical',
    subset='training',
    #This specifies that the generator should shuffle the data before generating batches.
    shuffle= True, 
    #This specifies the random seed to use for shuffling the data.
    seed= 1337
) 

#This part creates a generator for validation data using the flow_from_directory method of the ImageDataGenerator instance valid_datagen
valid_generator = valid_datagen.flow_from_directory(
    train_data_dir,
    target_size= (img_width, img_height),
    color_mode= 'rgb',
    batch_size= batch_size,  
    class_mode= 'categorical',
    subset='validation',
    shuffle= True, 
    seed= 1337
)

#This is a code snippet that extracts the labels from the training and validation generators, and converts them to one-hot encoded vectors using the to_categorical function from Keras.
#This extracts the number of classes from the training generator, which is stored in the class_indices attribute.
num_classes = len(train_generator.class_indices)  
#This extracts the class labels from the training generator, which are stored in the classes attribute.
train_labels = train_generator.classes 
#This converts the extracted class labels to one-hot encoded vectors using the to_categorical function, with the number of classes specified as num_classes.
train_labels = to_categorical(train_labels, num_classes=num_classes)
#This extracts the class labels from the validation generator.
valid_labels = valid_generator.classes 
valid_labels = to_categorical(valid_labels, num_classes=num_classes)
#This extracts the number of training and validationsamples from the training generator, which is stored in the filenames attribute.
nb_train_samples = len(train_generator.filenames)  
nb_valid_samples = len(valid_generator.filenames)

#This part creates an instance of the InceptionV3 model architecture from Keras, and prints a summary of the model.
InceptionV3 = applications.InceptionV3(include_top= False, input_shape= (img_width, img_height, channels), weights= 'imagenet')
InceptionV3.summary()


#This code snippet creates a new Sequential model by stacking layers from the pre-trained InceptionV3 model, 
#adding additional layers for fine-tuning, and compiling the model for training. 

#Initializes a new Sequential model.
model = Sequential()
#The loop iterates over all the layers in the InceptionV3 model and sets trainable=False for each layer. 
#This freezes the weights of the pre-trained layers so they are not updated during training.
for layer in InceptionV3.layers:
    layer.trainable= False
#     print(layer,layer.trainable)

#Adds the InceptionV3 base model with frozen weights to the new model.
model.add(InceptionV3)
#Adds a Global Average Pooling layer to reduce the spatial dimensions of the features.
model.add(GlobalAveragePooling2D())
#Adds a Dropout layer to prevent overfitting.
model.add(Dropout(0.2))
#Adds a Dense layer with 120 units and a softmax activation function for multi-class classification.
model.add(Dense(120,activation='softmax'))
#Prints a summary of the model architecture including the number of parameters in each layer.
model.summary()


#SVG(model_to_dot(model).create(prog='dot', format='svg'))
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)

#Compiles the model using the Adam optimizer with a learning rate of 0.0001, 
#categorical crossentropy loss function for multi-class classification, and accuracy as the evaluation metric.
model.compile(optimizer= keras.optimizers.Adam(lr= 0.0001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

#ModelCheckpoint callback from Keras, which is used to save the weights of the model at the epoch with the best validation loss during training. 
checkpoint = ModelCheckpoint(
    'baseline_model.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='auto',
    save_weights_only=False,
    period=1
)

#The EarlyStopping callback is used to prevent overfitting by stopping training when the validation loss has stopped improving for a certain number of epochs.
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=3,
    verbose=1,
    mode='auto'
)

#used to log training and validation metrics to a CSV file, which can be useful for monitoring the progress of training and for analyzing the performance of the model. 
csvlogger = CSVLogger(
    filename= "training_csv.log",
    separator = ",",
    append = False
)

#used to adjust the learning rate during training to help the model converge faster and potentially achieve better performance.
reduceLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1, 
    mode='auto'
)

callbacks = [checkpoint, earlystop, csvlogger,reduceLR]

#The fit method is used to train the model using the specified parameters and callbacks. 
#The method returns a History object that contains the training and validation metrics for each epoch, which can be used to analyze the performance of the model.
history = model.fit(
    #This is the generator that produces batches of training data.
    train_generator, 
    #his specifies the number of times the model will see the entire training dataset during training.
    epochs = 30,
    #his specifies the number of batches in one epoch.
    steps_per_epoch = nb_train_samples//batch_size,
    validation_data = valid_generator, 
    validation_steps = nb_valid_samples//batch_size,
    #This specifies that a progress bar will be displayed during training.
    verbose = 2, 
    callbacks = callbacks,
    #This specifies that the training data should be shuffled at the beginning of each epoch.
    shuffle = True
)

#The evaluate method is used to evaluate the performance of the model on a separate dataset, such as the validation dataset. 
#The method returns the loss and accuracy of the model on the evaluation dataset. 
(eval_loss, eval_accuracy) = model.evaluate(valid_generator, batch_size= batch_size, verbose= 1)
print('Validation Loss: ', eval_loss)
print('Validation Accuracy: ', eval_accuracy)

#This code snippet creates two plots to visualize the training and validation performance of the model over epochs.
plt.subplot()
plt.title('Model Accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training Accuracy','Validation Accuracy'])
plt.savefig('baseline_acc_epoch.png', transparent= False, bbox_inches= 'tight', dpi= 900)
plt.show()

plt.title('Model Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training Loss','Validation Loss'])
plt.savefig('baseline_loss_epoch.png', transparent= False, bbox_inches= 'tight', dpi= 900)
plt.show()
#The same set of commands is repeated for the 'Model Loss' plot, where the loss values are plotted instead of accuracy values. 
#These plots are commonly used to analyze the model's performance during training and to identify trends in accuracy and loss over epochs.
