from miniVGG import MiniVGGNet
from data.build_data import build_data
from utils.utils import display_bars, count_classes, scheduler
import tensorflow as tf 
import numpy as np 
from sklearn.model_selection import train_test_split
import os 
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
GPU = tf.config.list_physical_devices('GPU')
print(f"Available GPUs {GPU}")


augm_settings = {
    "vertical_flip": True, 
    "fill_mode": "nearest", 
    "rotation_range" : 30, 
    "width_shift_range" : 0.1, 
    "height_shift_range" : 0.1
}

config = {
    "epochs" : 30, 
    "batch_size" : 32, 
    "lr" : 0.01, 
    "image_size" : 224
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

#dimensions 
print(f"Data shape : {data.shape}")
print(f"Labels shape : {labels.shape}")
num_classes = len(target_names)
w, h, c = data.shape[1:]

# anaylising imbalanceness
# reports = count_classes(labels, target_names)
# display_bars(
#     [reports[classe] for classe in reports], 
#     target_names
# )

# data splitting 
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.30, shuffle=True)
X_train = X_train / 255.0 
X_test = X_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
# data augmentation

data_augm_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    **augm_settings
)

input_shape = X_train.shape[1:]
# resnet50 : transfer learning
base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        classes = num_classes, 
        weights='imagenet', 
        input_shape=(200, 200, 3)
    )   
print(base_model.summary())
def resnet50_transfer(input_shape): 
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.image.resize(inputs, (config["image_size"], config["image_size"]),preserve_aspect_ratio=True)
    base_model.trainable = False
    # ---- top model ---- 
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = resnet50_transfer(input_shape)
print(model.summary())

# ---- training the top model ------ #

# callbacks 
log_dir = f"./logs/fit/resnet_transfered_{config['batch_size']}_ep{config['epochs']}_shuffled"
checkpoints_path = "./train/resnet_transfered_.h5"
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(checkpoints_path, versbose=1), 
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
] 
optimizer =  tf.keras.optimizers.SGD(config["lr"] * (10**-3)) # very low learning rate 
model.compile(
    optimizer=optimizer, 
    loss="categorical_crossentropy", 
    metrics=["accuracy"], 
)
# h = model.fit(
#     data_augm_gen.flow(X_train, y_train, batch_size=config["batch_size"]), 
#     steps_per_epoch=len(X_train) // config["batch_size"],
#     epochs=config["epochs"], 
#     verbose=1, 
#     validation_data=(X_test, y_test), 
#     callbacks=callbacks
# )

# laod weight  
model.load_weights(checkpoints_path)

predictions = model.predict(X_test) 
print(predictions) 
print(y_test)

print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=target_names))


print("Fine Tuning Process")
print("-"*50)
# unfreezing the layers 
base_model.trainable = True
#optimizer
optimizer =  tf.keras.optimizers.SGD(config["lr"] * 10**-3)
log_dir = f"./logs/fit/resnet_transfered_{config['batch_size']}_ep{config['epochs']}_shuffled_fined_tuned"
checkpoints_path = "./train/resnet_transfered_fine_tuned.h5"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoints_path, versbose=1), 
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
] 

# re-compiling
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(data_augm_gen.flow(X_train, y_train, batch_size=config["batch_size"]), epochs=10, callbacks=[callbacks], validation_data=(X_test, y_test))
model.load_weights(checkpoints_path)

predictions = model.predict(X_test) 
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=target_names))
