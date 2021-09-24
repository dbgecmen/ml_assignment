from rnn_model import train_and_plot_best_rnn, grid_search_rnn
from rf_model import train_and_plot_best_rf, grid_search_rf

if __name__ == '__main__':

    """
    Functions below train models and plot results using best hyper-parameters found during the grid search 
    
    For both models the confusion matrix is plotted and saved in /figures
    For the RNN models also the training and validation loss and accuracy is plotted vs epochs
    
    "balanced" determines whether class weighting is used, if True class weighting is applied
    """

    # plots results of best rnn
    train_and_plot_best_rnn(balanced=True)

    # plots results of best rnn using class_weighting
    train_and_plot_best_rnn(balanced=True)

    # plots results of best random forest
    train_and_plot_best_rf(balanced=True)

    # plots results of best random forest using class_weighting
    train_and_plot_best_rf(balanced=False)

    """
    Functions below perform grid search to find best hyper-parameters for either Random Forest or RNN classifier
    After completion the results are plotted 
    
    "balanced" determines whether class weighting is used, if True class weighting is applied
    
    To modify the search space of the grid see within the functions grid_search_rnn and grid_search_rf
    
    Note that grid search can take vary long to complete
    """

    # performs grid search for rnn
    grid_search_rnn(balanced=False)

    # performs grid search for rnn using class_weighting
    grid_search_rnn(balanced=True)

    # performs grid search for random forest
    grid_search_rf(balanced=False)

    # performs grid search for random forest using class_weighting
    grid_search_rf(balanced=True)




    


    

