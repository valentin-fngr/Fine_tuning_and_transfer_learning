import matplotlib.pyplot as plt 
import numpy as np
import numpy as np

def display_bars(heights, target_names): 
    plt.figure(figsize=(10,10))
    plt.bar(target_names, heights, linewidth=1)
    plt.title("Imbalanceness Analysis")
    plt.xticks(rotation=90)
    return plt.show()

def count_classes(labels, target_names): 
    unique, counts = np.unique(labels, return_counts=True)
    z = dict(zip([target_names[val] for val in unique], counts)) 

    return z
    

def scheduler(epoch, lr): 
    if epoch < 30: 
        return lr
    return lr * np.exp(-0.1)
    

    