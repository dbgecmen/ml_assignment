from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, GRU
from keras.wrappers.scikit_learn import KerasClassifier

import pickle
from preprocessing import load_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from result_processing import process_predictions, set_up_confusion_matrix, plot_matrix
import matplotlib.pyplot as plt

def save_pickle(path ,data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def build_rnn(output_dim, rnn_layer_type, rnn_layer_units, depth, optimizer):

    model = Sequential()
    embedding_layer = Embedding(input_dim=34, output_dim=output_dim,  input_length=8)
    model.add(embedding_layer)
    model.add(rnn_layer_type(units = rnn_layer_units))
    for i in range(1, depth):
        model.add(rnn_layer_type(units = rnn_layer_units))
    model.add(Dense(units = 33))
    model.add(Activation('softmax'))
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['acc'])   
    return model

def train_and_plot_best_rnn(balanced=False):
    """
    Loads best params and constructs model. If saved weights are available those are loaded and training is skipped.
    if best params are not available the function stops and a grid search must be ran first.
    """

    if balanced:
        filename = 'rnn_class_weight'
    else:
        filename = 'rnn_baseline'

    X_train, y_train, X_test, y_test, class_weights, label_dict = load_data(onehot_labels=True)

    try:
        params = load_pickle(f'results/best_params_{filename}.p')
    except FileNotFoundError:
        print('Best parameters are not saved, first run the grid search')
        return

    model = build_rnn(params['output_dim'], params['rnn_layer_type'], params['rnn_layer_units'], params['depth'],
                      params['optimizer'])
    try:
        history = load_pickle(f'results/train_history_{filename}.p')
        model.load_weights(f'results/model_{filename}.h5')
    except:
        print('Best model state or training history not saved, training and saving model')
        history = model.fit(X_train, y_train,
                            epochs=params['epochs'],
                            batch_size=params['batch_size'],
                            validation_split=0.2)
        history = history.history
        model.save(f'results/model_{filename}.h5')
        save_pickle(f'results/train_history_{filename}.p', history)

    y_hat_raw = model.predict(X_test)
    y_hat = process_predictions(y_hat_raw)

    acurracy = accuracy_score(y_test, y_hat)
    c_mat = set_up_confusion_matrix(y_test, y_hat, normalize=True)

    plot_matrix(c_mat, label_dict, accuracy=acurracy, filename=filename)

    fig = plt.figure()
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('RNN model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'figures/model_accuracy_{filename}.png', bbox_inches='tight')

    fig = plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('RNN model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'figures/model_loss_{filename}.png', bbox_inches='tight')

def grid_search_rnn(balanced=False):
    """
    Performs grid search to find best RNN hyper-parameters

    if "balanced" is true, class weighting is applied
    """

    if balanced:
        filename = 'rnn_class_weight'
    else:
        filename = 'rnn_baseline'

    # #Define parameters
    optimizer = ['rmsprop', 'adam']
    batch_size = [32, 64, 128, 256]
    epochs = [30, 50, 100, 200]
    depth = [0, 1]
    rnn_layer_type = [GRU, LSTM]
    rnn_layer_units = [8, 16, 32, 64]
    output_dim = [32]

    X_train, y_train, X_test, y_test, class_weights, label_dict = load_data(onehot_labels=True)

    param_grid = dict(optimizer=optimizer,
                      epochs=epochs,
                      batch_size=batch_size,
                      depth=depth,
                      rnn_layer_type=rnn_layer_type,
                      rnn_layer_units=rnn_layer_units,
                      output_dim=output_dim)

    model = KerasClassifier(build_fn=build_rnn)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

    if balanced:
        grid_result = grid.fit(X_train, y_train, verbose=0, class_weight=class_weights)
    save_pickle(f'results/grid_result_{filename}.p', grid_result.cv_results_)
    save_pickle(f'results/best_params_{filename}.p', grid_result.best_params_)

    # plot best results
    train_and_plot_best_rnn(balanced=balanced)

if __name__ == '__main__':

    train_and_plot_best_rnn(balanced=False)
    train_and_plot_best_rnn(balanced=True)

    # grid_search_rnn(balanced=True)