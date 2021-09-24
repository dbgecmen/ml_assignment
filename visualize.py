import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from keras.preprocessing.text import Tokenizer
from preprocessing import load_data_train, load_data_test

def visualize_class_distribution(data, title, path):
    t = Tokenizer()
    t.fit_on_texts(data)
    a= dict(t.word_counts)
    df = pd.DataFrame(a.items(), columns = ['Character', '# Samples']) 
    df = df.sort_values(by=['# Samples'], ascending=False)
    ax = df.plot.bar(x='Character', y='# Samples', title = title, rot=0)
    fig = ax.get_figure()
    fig.savefig(path)
    plt.close()
    return

if __name__ == '__main__':
    """
    Plot distribution of classes in the train and test labels.
    """
    train_data = load_data_train()
    test_data = load_data_test()  
    y_train = np.array([lst[-1] for lst in train_data])
    y_test = np.array([lst[-1] for lst in test_data])
    visualize_class_distribution(y_train, "Labels Distribution Train", 'figures/train_labels_distribution.png')
    visualize_class_distribution(y_test, "Labels Distribution Test", 'figures/test_labels_distribution.png')    

