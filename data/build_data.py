import os 
import numpy as np 
import tensorflow as tf
import pickle

'''
    extracting and saving data
'''


DIRECTORY = os.path.join(os.getcwd(), "101_ObjectCategories")
def build_data(): 
    
    reports = {}
    data = [] 
    labels = []
    target_names = [name for name in os.listdir(os.path.join(DIRECTORY))]

    print(target_names)
    gen_counter = 0 
    for classe in os.listdir(DIRECTORY): 
        reports[classe] = {}
        print(f"Loading data from {classe} ... \n ")
        # open directory 
        classe_path = os.path.join(DIRECTORY, classe) 
        counter = 0
        for image in os.listdir(classe_path): 
            image_path = os.path.join(classe_path, image)
            try:
                img_file = tf.keras.preprocessing.image.load_img(image_path, color_mode='rgb', interpolation='nearest') 
                img_array = tf.keras.preprocessing.image.img_to_array(img_file) 
                data.append(img_array)
                labels.append(target_names.index(classe))
                counter += 1
            except Exception as e: 
                print("something went wrong ")
                print(e) 
        
        # reports
        reports[classe]["size"] = counter

        gen_counter += counter
        print(f"loaded {counter} images from : {classe} \n")
    print(f"Loaded a total of : {gen_counter} images \n")

    for classe in reports: 
        reports[classe]["freq"] = round(reports[classe]["size"] / gen_counter, 2)

    # serializing
    
    labels = np.array(labels).reshape(-1, 1)
    with open('data.pickle', 'wb') as f: 
        pickle.dump({
            "data" : data, 
            "labels" : labels, 
            "target_names" : target_names, 
            "reports": reports
        }, f)
        print("Successfully serialized the data !") 
    
    
build_data()