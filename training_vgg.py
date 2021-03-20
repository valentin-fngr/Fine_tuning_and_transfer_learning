from miniVGG import MiniVGGNet
from data.build_data import build_data
from utils.utils import display_bars
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
# retrieving data : load serialized data or build it
if os.path.isfile("data/data.npy"):
    with open("./data/data.npy", "rb") as f: 
        print("Loading serialized data ... \n")
        serialized_data = pickle.load(f)
        data = np.array(serialized_data["data"])
        labels = serialized_data["labels"]
        target_names = serialized_data["target_names"], 
        reports = serialized_data["reports"]
else: 
    print("Something went wrong ")
    data_infos = build_data() # no serializing
    data = np.array(data_infos["data"])
    labels = data_infos["labels"]
    target_names = data_infos["target_names"], 
    reports = data_infos["reports"] 

#dimensions 
print(f"Data shape : {data.shape}")
print(f"Labels shape : {labels.shape}")
num_classes = len(reports)
w, h, c = data.shape[1:]

# anaylising imbalanceness

display_bars(
    [reports[classe]["size"] for classe in reports], 
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

# callbacks 
log_dir = f"./logs/fit/miniVGG_bs{config[batch_size]}_ep{config[epochs]}"
checkpoints_path = "./tmp/checkpoint"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoints_path, versbose=1), 
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
] 


# Loading the model  
model = MiniVGGNet.build(w, h, c)

# compile
lr = 0.01
optimizer =  tf.keras.optimizers.SGD(lr)

model.compile(
    optimizer=config["optimizer"], 
    loss="categorical_cross_entropy", 
    metrics=["accuracy"], 
)

# fit data 

model.fit(
    data_augm_gen.flow(X_train, y_train, batch_size=config["batch_size"]), 
    steps_per_epoch=len(X_train) // config["batch_size"],
    epochs=config["epochs"], 
    verbose=1, 
    validation_split=(X_test, y_test)
)
