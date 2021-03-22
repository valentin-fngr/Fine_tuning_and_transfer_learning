from miniVGG import MiniVGGNet
from data.build_data import build_data
from utils.utils import display_bars, count_classes, scheduler
import tensorflow as tf 
import numpy as np 
from sklearn.model_selection import train_test_split
import os 
import matplotlib.pyplot as plt
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

GPU = tf.config.list_physical_devices('GPU')
print(f"Available GPUs {GPU}")


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
    "lr" : 0.01, 
    "image_size": 150
}
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
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, shuffle=True)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# examples 
plt.figure(figsize=(10, 10))
images_show = X_train[:9] 
labels_show = y_train[:9] 
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images_show[i].astype(np.uint8))
    plt.title(f"{np.argmax(labels_show[i])} : {target_names[np.argmax(labels_show[i])]}")
    plt.axis("off")
plt.show()

# data augmentation

data_augm_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0, 
    horizontal_flip=True, 
    vertical_flip=True, 
    rotation_range=25,
    zoom_range=0.3, 
    shear_range=0.2, 
    width_shift_range=0.1, 
    height_shift_range=0.1
)

# callbacks 
log_dir = f"./logs/fit/miniVGG_bs{config['batch_size']}_ep{config['epochs']}"
checkpoints_path = "./train/best_model_vgg.h5"
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(checkpoints_path, versbose=1), 
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
] 


# Loading the model  
model = MiniVGGNet.build(w, h, c, num_classes)
print(model.summary())
# compile
lr = 0.01
optimizer =  tf.keras.optimizers.SGD(lr)
model.compile(
    optimizer=optimizer, 
    loss="categorical_crossentropy", 
    metrics=["accuracy"], 
)
#fit data 
h = model.fit(
    data_augm_gen.flow(X_train, y_train, batch_size=config["batch_size"]), 
    steps_per_epoch=len(X_train) // config["batch_size"],
    epochs=config["epochs"], 
    verbose=1, 
    validation_data=(X_test, y_test), 
    callbacks=callbacks
)

