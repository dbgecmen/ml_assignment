import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.utils.class_weight import compute_class_weight

# Import sequences of raw datasets 
def load_data_train():
    data = []
    with open('data/train.csv', 'r') as f:
        for line in f.read().splitlines():
            result = ' '.join(line)
            data.append(result)
    return data

def load_data_test():
    data = []
    with open('data/answers.csv', 'r') as f:
        for line in f.read().splitlines():
            result = ' '.join(line)
            data.append(result)
    return data

def load_data_hidden():
    hidden = []
    with open('data/hidden.csv', 'r') as f:
        for line in f.read().splitlines():
            result = ' '.join(line[:-1])
            hidden.append(result)
    return hidden   


# Get train and test data for RNN 
def load_data(onehot_labels=True):
    train_data = load_data_train()
    test_data = load_data_test()

    # prepare dataset of input to output pairs encoded as integers
    t = Tokenizer()
   
    # Create index based on character frequency
    t.fit_on_texts(train_data)
   
    # Assign integers to corresponding characters
    train = t.texts_to_sequences(train_data)
    test = t.texts_to_sequences(test_data)

    X_train = np.array([lst[:-1] for lst in train])
    X_test = np.array([lst[:-1] for lst in test])

    # The problem is a sequence classification task, where each of the 33 characters represent a different class.
    # We can convert the output to a one hot encoding, using the Keras built-in to_categorical()
    # one hot encode the output variable
    if onehot_labels:
        y_train = np_utils.to_categorical([lst[-1] for lst in train], num_classes=33)
        y_test = np_utils.to_categorical([lst[-1] for lst in test], num_classes=33)
        # compute class weights:
        y_integers = [lst[-1] for lst in train]
        class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)

        # insert zeros in class weight array to account for classes not present in the training labels
        label_frequency = np.sum(y_train, axis=0)
        class_weights_full = np.zeros(label_frequency.shape)
        class_weights_full[label_frequency > 0] = class_weights
        d_class_weights = dict(enumerate(class_weights_full))
    else:
        y_train = [lst[-1] for lst in train]
        y_test = [lst[-1] for lst in test]

        # Not one hot labels only used with random forest, which computes class_weights itself.
        d_class_weights = None

    return X_train, y_train, X_test, y_test, d_class_weights, t.word_index


if __name__ == '__main__':
    load_data()