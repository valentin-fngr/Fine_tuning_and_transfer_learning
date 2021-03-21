from miniVGG import MiniVGGNet
from data.build_data import build_data
from utils.utils import display_bars, count_classes
import tensorflow as tf 
import numpy as np 
from sklearn.model_selection import train_test_split
import os 
import matplotlib.pyplot as plt
import pickle

augm_settings = {
    "vertical_flip": True, 
    "horizontal_flip": True, 
    "rescale" : 1/255.0, 
    "fill_mode": "nearest", 
    "rotation_range" : 30, 
    "width_shift_range" : 0.1, 
    "height_shift_range" : 0.1
}

config = {
    "epochs" : 50, 
    "batch_size" : 32, 
    "lr" : 0.01
}


# build top 
def build_top(input_shape): 
    inputs = tf.keras.Input(shape=input_shape)

    # todo 
    # ADD FILTERS AND ALL !! 

    x  = tf.keras.layers.Conv2D(
        input_shape=input_shape, 
        activation="relu", 
        padding="same"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x  = tf.keras.layers.Conv2D(
        input_shape=input_shape, 
        activation="relu", 
        padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    

# retrieving data : load serialized data or build it

if os.path.isfile("data/data.npy"):
    with open("./data/data.npy", "rb") as f: 
        print("Loading serialized data ... \n")
        data = np.load(f) 
        labels = np.load(f)
        target_names = np.load(f)
else: 
    print("Something went wrong ")
    
    data_infos = build_data() # no serializing
    data = data_infos["data"]
    labels = data_infos["labels"]
    target_names = data_infos["target_names"], 

#dimensions 
print(f"Data shape : {data.shape}")
print(f"Labels shape : {labels.shape}")
num_classes = len(target_names)
w, h, c = data.shape[1:]

# anaylising imbalanceness
reports = count_classes(labels, target_names)
display_bars(
    [reports[classe] for classe in reports], 
    target_names
)

# data splitting 
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
# data augmentation

data_augm_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    **augm_settings
)


# resnet50
base_model = tf.keras.applications.ResNet50V2(
    include_top=False,
    classes = num_classes, 
    weights='imagenet', 
    input_shape=(300, 200, 3)
)
base_model.trainable = False
print(base_model.summary)
