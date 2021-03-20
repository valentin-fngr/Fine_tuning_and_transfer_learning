import matplotlib.pyplot as plt 

def display_bars(heights, target_names): 
    plt.figure(figsize(10,10))
    plt.bar(target_names, heigths, linewidth=1)
    plt.title("Imbalanceness Analysis")
    return plt.show()