import pickle
from sklearn.ensemble import RandomForestClassifier
from keras.utils import np_utils
from preprocessing import load_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from result_processing import set_up_confusion_matrix, plot_matrix
import joblib

def save_pickle(path ,data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def train_and_plot_best_rf(balanced=True):
    """
    Loads best params and constructs model. If saved weights are available those are loaded and training is skipped.
    if best params are not available the function stops and a grid search must be ran first.
    """

    if balanced:
        filename = 'rf_class_weight'
    else:
        filename = 'rf_baseline'

    X_train, y_train, X_test, y_test, _, label_dict = load_data(onehot_labels=False)

    try:
        model = joblib.load(f'results/model_{filename}')
    except FileNotFoundError:
        print('Best model state not saved, loading best params, training and saving model')

        try:
            params = load_pickle(f'results/best_params_{filename}.p')
        except FileNotFoundError:
            print('Best parameters are not saved, first run the grid search')
            return

        best_n_estimators = params['n_estimators']
        best_max_depth = params['max_depth']
        best_criterion = params['criterion']
        best_min_samples_leaf = params['min_samples_leaf']
        best_max_leaf_nodes = params['max_leaf_nodes']
        best_bootstrap = params['bootstrap']

        # # evalutaion of individual RF classifiers: plot confusion matrix
        if balanced:
            model = RandomForestClassifier(n_estimators=best_n_estimators,
                                           max_depth=best_max_depth,
                                           criterion=best_criterion,
                                           min_samples_leaf=best_min_samples_leaf,
                                           max_leaf_nodes=best_max_leaf_nodes,
                                           bootstrap=best_bootstrap,
                                           class_weight='balanced')
        else:
            model = RandomForestClassifier(n_estimators=best_n_estimators,
                                           max_depth=best_max_depth,
                                           criterion=best_criterion,
                                           min_samples_leaf=best_min_samples_leaf,
                                           max_leaf_nodes=best_max_leaf_nodes,
                                           bootstrap=best_bootstrap)

        history = model.fit(X_train, y_train)

        joblib.dump(model, f'results/model_{filename}')

    y_hat = model.predict(X_test)

    # the c_mat function requires one hot encoded labels so transform the labels:
    y_hat_oh = np_utils.to_categorical(y_hat, num_classes=33)
    y_test_oh = np_utils.to_categorical(y_test, num_classes=33)

    accuracy = accuracy_score(y_test, y_hat)

    c_mat = set_up_confusion_matrix(y_test_oh, y_hat_oh)

    plot_matrix(c_mat, label_dict, accuracy=accuracy, filename=filename)




def grid_search_rf(balanced=False):
    """
    Performs grid search to find best RNN hyper-parameters

    if "balanced" is true, class weighting is applied

    if memory issues arise, modify n_jobs of GridSearchCV to use less cores (-1 means using all cores)
    """

    X_train, y_train, X_test, y_test, _, label_dict = load_data(onehot_labels=False)

    n_estimators = [50, 100, 500, 1000, 2000, 5000]
    max_depth = [5, 10, 30, None]
    criterion = ['gini', 'entropy']
    min_samples_leaf = [1, 5, 10, 20]
    max_leaf_nodes = [5, 10, None]
    bootstrap = [True, False]

    param_grid = dict(n_estimators=n_estimators,
                      max_depth=max_depth,
                      criterion=criterion,
                      min_samples_leaf=min_samples_leaf,
                      max_leaf_nodes=max_leaf_nodes,
                      bootstrap=bootstrap)

    if balanced:
        filename = 'rf_class_weight'
        model = RandomForestClassifier(class_weight='balanced')
    else:
        filename = 'rf_baseline'
        model = RandomForestClassifier()

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

    grid_result = grid.fit(X_train, y_train)

    save_pickle(f'results/grid_result_{filename}.p', grid_result.cv_results_)
    save_pickle(f'results/best_params_{filename}.p', grid_result.best_params_)

    train_and_plot_best_rf(balanced=balanced)

if __name__ == '__main__':

    # to plot best random forest with
    train_and_plot_best_rf(balanced=True)
    train_and_plot_best_rf(balanced=False)

    # grid_search_rf(balanced=False)