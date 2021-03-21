import os 
import numpy as np 
import tensorflow as tf
import pickle

'''
    extracting and saving data
'''

DIRECTORY = os.path.join("data", "101_ObjectCategories")
def build_data(serialize=False): 
    
    data = [] 
    labels = []
    target_names = [name for name in os.listdir(os.path.join(DIRECTORY))]

    print(target_names)
    gen_counter = 0 
    for classe in os.listdir(DIRECTORY): 
        print(f"Loading data from {classe} ... \n ")
        # open directory 
        classe_path = os.path.join(DIRECTORY, classe) 
        counter = 0
        for image in os.listdir(classe_path): 
            image_path = os.path.join(classe_path, image)
            try:
                img_file = tf.keras.preprocessing.image.load_img(image_path, color_mode='rgb', interpolation='nearest') 
                img_array = tf.keras.preprocessing.image.img_to_array(img_file).astype(np.uint8)
                img_array = tf.image.resize(img_array, [200, 200])
                data.append(img_array)
                labels.append(target_names.index(classe))
                counter += 1
            except Exception as e: 
                print(e) 
                print("something went wrong ")
        
        gen_counter += counter
        print(f"loaded {counter} images from : {classe} \n")
    print(f"Loaded a total of : {gen_counter} images \n")
    
    # serializing
    data = np.array(data)
    labels = np.array(labels).reshape(-1, 1)

    print(data.shape)

    if not os.path.isfile("./data/data.npy"):
        print("opening folder ! ")
        with open('./data/data.npy', 'wb') as f: 
            print("serializing ... \n")
            np.save(f, data), 
            np.save(f, labels) 
            np.save(f, np.array(target_names))
            print("Successfully serialized the data !") 

    return {
        "data" : np.array(data), 
        "labels" : np.array(labels), 
        "target_names" : target_names 
    }
    