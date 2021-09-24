import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def process_predictions(y_pred):
    maxpos = np.argmax(y_pred, axis=1)
    y_hat_classes = np.zeros(y_pred.shape)
    y_hat_classes[np.arange(y_pred.shape[0]), maxpos] = 1
    return y_hat_classes

def set_up_confusion_matrix(y_true, y_pred, normalize=True):
    """
    outputs the multilabel confusion matrix

    assumes the data is presented: [n_samples, n_classes]
    """
    n_class = y_true.shape[1]
    c_mat = np.zeros((n_class, n_class))

    for ii in range(n_class):
        y_true_class = y_true[:, ii]
        y_pred_filtered = y_pred[y_true_class == 1]
        c_mat[:, ii] = np.sum(y_pred_filtered, axis=0)

    if normalize:
        n_samples = np.sum(c_mat, axis=0)
        filtered_c_mat = c_mat[:, n_samples > 0] # to avoid dividing by zero
        filtered_n_samples = n_samples[n_samples > 0]
        filtered_normalized = filtered_c_mat/filtered_n_samples
        c_mat[:, n_samples > 0] = filtered_normalized

    return c_mat

def plot_matrix(c_mat, label_dict, accuracy=None, f1score=None, filename=''):
    """
    plots and saves confusion matrix, accuracy or F1 score is displayed in title if passed
    """

    # process classes into label list
    labels = [key for key in label_dict]
    # Tokenizer shifts the labels by one index, that is corrected below
    labels.insert(0, labels.pop())

    df_c_mat = pd.DataFrame(c_mat, index=labels, columns=labels)


    # make title string to display accuracy and f1 score, if passsed:
    title = ''
    if accuracy:
        title += f'Accuracy = {accuracy} '
    if f1score:
        title += f'F1-score = {f1score} '

    plt.figure(figsize = (20,20))
    sn.heatmap(df_c_mat,
               annot=True,
               annot_kws={"size": 10},
               square=True,
               cmap="YlGnBu",
               linewidths=.5,
               vmin=0,
               vmax=1)
    if accuracy or f1score:
        plt.title(title)
    plt.savefig(f'figures/confusion_matrix_{filename}.png', bbox_inches='tight')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
