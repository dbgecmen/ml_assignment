
**[Introduction](#introduction)**                                          |
**[Install Requirements](#install-requirements)**                          |
**[Data Preprocessing](#data-preprocessing)**                              |
**[Models](#models)**                                                      | 
**[Results](#results)**                                                    |
**[Conclusions and Recommendations](#conclusions-and-recommendations)**                    
---

# Introduction

Two AI chatbots had been instructed to work out how to negotiate between themselves, but they were not told to use comprehensible English, allowing them to create their own "shorthand" version of the language, which by first observations has quite simple grammar and consist of only three rules:

1. Alphabet consists of 33 characters: 26 upper case letters and 7 digits: `ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567`.
2. A sentence always consists of 9 characters from the chat bot alphabet, never less and never more.
3. Chat bots never pronounce the same sentence more than once, hence there are no duplicates in their conversation.

To reverse engineer the chat bot language, we started to continuously record their conversations. Here are a couple examples of chat bot sentences collected:
- RSWW1KS2S
- 6LZTVD111
- 6M2MMMMMM

After a while the bots noticed that we were saving their conversation, so they replaced the last character with a "?". Here are a couple examples of the newly collected sentences:
- R2SR2SR2'?'
- 2222MS22'?'
- M22M22MS'?'

We need to recover the last hidden characters! 

# Install Requirements
Note that you need Python 3.6–3.8 for TensorFlow. I've used Python 3.8.

Run `pip install -r requirements.txt` to install the necessary packages. If issues arise try `conda install --file requirements.txt`.

# Data Pre-processing 

The data consists of three sets:
- Train: split into train and validation sets.
- Answers (test): unseen data to assess the performance of the final model.
- Hidden: data containing test sentences where the last character is replaced by a question mark `?`.

One sentence consists of 33 characters including A-Z and 1-7. The models that we build take and input sequence length of 8 characters and output one character. 

The characters are encoded as integers. For the output character there are two different represenations:

* Encoded as an integer.
* One-hot encoded and represented as a `[1,33]` dimensional array.

For both the train and test set the label distribution is plotted:

![Distribution of labels in train](https://github.com/dbgecmen/ml-assignment/blob/main/figures/train_labels_distribution.png?raw=true)

![Distribution of labels in test](https://github.com/dbgecmen/ml-assignment/blob/main/figures/test_labels_distribution.png?raw=true)

There is a severe class imbalance. It is known that classifiers do not perform well on imbalanced datasets. They can make accurate classifications for the majority classes at expense of the minority classes. This will result in a high classification accuracy, which is a bit misleading. Therefore, these type of problems are also known as the accuracy paradox. For these type of problems, the accuracy is a misleading measure. Additional measures are required to evaluate a classifier, such as the confusion matrix. 

|    Labels     | # Classes     | 
| ------------- | ------------- | 
|  Train        | 31            | 
|  Test         | 13            | 

From the previous Table and Figures, we can see that the number of classes to predict for test is low and that these classes belong to the majority classes from train. 

# Models

The problem is a multiclass classification on imbalanced data. To work towards a solution for the problem, we will focus on two different models: Recurrent Neural Networks (RNN) and Random Forests (RF). 

## RNN

In practice, commonly Recurrent Neural Networks (RNNs) are used to solve these type of problems, due to good performance with sequential data. There are different type of RNN architectures. The Vanilla RNN suffers from vanishing and exploding gradients. Two very succesfull RNN architectures that deal with vanishing gradients are the Long short-term memory (LSTM) and the Gated recurrent unit (GRU). Both architectures use multiplicative elementwise gates together with additive interactions to avoid the vanishing gradient problem. 

Note that we will not tweak the parameters in the cell as stated in their original papers. It is proven that tweaks tried were not significantly better than the original LTSM for a large number of problems [1].

## Decision Tree (Random Forest)

Besides the RNN we will also look into the Random Forest, as it is known it works well with imbalanced data. 

# Results

## RNN

We will start by developing a simple RNN network to learn the last character in a sequence of characters in a sentence. The first model is an RNN model and will be used as a baseline for further improvements. 

For the loss the Categorical Cross-Entropy is used, as we have a multi-class classification problem. 5-fold cross validation is used to evaluate the performance of the model.

The hyperparameters for grid search are given in the table below.

|  Parameters   |  Grid           | 
| ------------- | -------------   |
|  Batch        |  [32 64 128 256]|   
|  Epochs       |  [30 50 100 200]|  
|  Optimizer    |  [RMSprop, Adam]|  
|  RNN variant  |  [LSTM, GRU]    |  
|  Units        |  [8 16 32 64]   |  
|  Output dim   |  [32]           |  

The best configuration of parameters is given in the Table below.

|  Parameters   |  Value        | 
| ------------- | ------------- |
|  Batch        |  128          |   
|  Epochs       |  100          |  
|  Optimizer    |  Adam         |  
|  RNN variant  |  GRU          |  
|  Depth        |  1            |  
|  Units        |  64           |  
|  Output dim   |  32           | 

In the Figures below, the accuracy and loss are plotted for the train and validation set. 

![RNN Train and Test Accuracy](https://github.com/dbgecmen/ml-assignment/blob/main/figures/model_accuracy_rnn_baseline.png?raw=true)

![RNN Train and Test Loss](https://github.com/dbgecmen/ml-assignment/blob/main/figures/model_loss_rnn_baseline.png?raw=true)

The test accuracy is 79%, which is higher than the train accuracy. This is because the model performs well in the particular group of data present in the test data.

The confusion matrix is given below. The x-axis represents the true labels and the y-axis the predicted labels. On the diagonal we can get an idea of the accuracy of predicting each class. 

From the confusion matrix we can see that there is a higher classification of samples from the majority classes. Unlike the minority classes. Suprisingly, the minority class `a` is predicted by the model.  

![RNN Confusion Matrix](https://github.com/dbgecmen/ml-assignment/blob/main/figures/confusion_matrix_rnn_baseline.png?raw=true)

We are going to apply cost-sensitive learning assigning higher class weights to minority class instances and lower weights for classes with more instances. 

The hyperparameters for grid search are the same as previously. The best configuration of parameters for class weighted RNN is given in the Table below. 

|  Parameters   |  Value        | 
| ------------- | ------------- |
|  Batch        |  64           |   
|  Epochs       |  100          |  
|  Optimizer    |  Adam         |  
|  RNN variant  |  [LSTM, GRU]  |  
|  Depth        |  1            |  
|  Units        |  64           |  
|  Output dim   |  32           | 

In the Figures below, the accuracy and loss are plotted for the train and validation set. 

![Weighted Classes RNN Train and Validation Accuracy](https://github.com/dbgecmen/ml-assignment/blob/main/figures/model_accuracy_rnn_class_weight.png?raw=true)

![Weighted Classes RNN Train and Validation Loss](https://github.com/dbgecmen/ml-assignment/blob/main/figures/model_loss_rnn_class_weight.png?raw=true)

![Weighted Classes RNN Confusion Matrix](https://github.com/dbgecmen/ml-assignment/blob/main/figures/confusion_matrix_rnn_class_weight.png?raw=true)

We see a slight improvement in the validation accuracy when using class weighting compared to the RNN that did not use class weights. Indicating that the network generalizes slightly better. Unfortunately also the difference between the train and validation accuracy is larger, indicating the RNN is sightly overfitted. 

Comparing the confusion matrix of the cost-sensitive RNN with the confusion matrix of the baseline RNN the model using class-weighting seems to perform worse. The overall accuracy is lower, and the accuracy for individual classes has also reduced. For example the accuracy of predicting the letter `s` has dropped from 88% to 58%, indicating that the class weighting is working, because predicting the majority class `s` is now considered less important. 

Because of the higher validation accuracy using cost-sensitive learning one would expect the test accuracy to improve, which is not the case. Also it is expected that the test accuracy would be about the same as the validation accuracy, while in reality the test accuracy is much higher than the validation accuracy. This is because the test set almost exclusively contains the majority labels, which the baseline RNN can predict very well due to the large amount of samples it is trained with. Because the class-weighted RNN is trained to assign less value to predicting these majority classes correctly, it is understandable that the accuracy drops when predicting a test set containing almost exclusively majority labels. 

## Random Forest

The hyperparameters for grid search are given in the Table below.

|  Parameters       |  Grid                           | 
| -------------     | -------------                   |
|  Max Depth        | [5, 10, 30, None]               |   
|  # Trees          | [50, 100, 500, 1000, 2000, 5000]|  
|  Criterion        | [gini, entropy]                 | 
|  Min Samples Leaf | [1, 5, 10, 20]                  | 
|  Max Leaf Nodes   | [5, 10, None]                   | 
|  Bootstrap Samples| [True, False]                   | 

The best configuration of parameters is given in the Table below.

|  Parameters       |  Grid         | 
| -------------     | ------------- |                   
|  Max Depth        | 10            |
|  # Trees          | 5000          |
|  Criterion        | entropy       |          
|  Min Samples Leaf | 10            |       
|  Max Leaf Nodes   | None          |          
|  Bootstrap Samples| True          |           

![RF Confusion Matrix](https://github.com/dbgecmen/ml-assignment/blob/main/figures/confusion_matrix_rf_baseline.png?raw=true)

The test accuracy is 75% and the validation accuracy is 40%.

We are going to apply cost-sensitive learning assigning higher class weights to minority class instances and lower weights for classes with more instances. 

The hyperparameters for grid search are the same as previously. The best configuration of parameters for class weighted RF is given in the Table below. 

|  Parameters       |  Grid         | 
| -------------     | ------------- |                   
|  Max Depth        | 10            |
|  # Trees          | 2000          |
|  Criterion        | entropy       |          
|  Min Samples Leaf | 1             |       
|  Max Leaf Nodes   | Nome          |          
|  Bootstrap Samples| True          | 

![Weighted Classes RF Confusion Matrix](https://github.com/dbgecmen/ml-assignment/blob/main/figures/confusion_matrix_rf_class_weight.png?raw=true)

The test accuracy is 64 % and the validation accuracy is 34%.

From the confusion matrices for the baseline RF and the weighted RF, we can see that adding class weights dropped the test and validation accuracy. The effect is similar to the described result when adding class weights to the RNN, the predictions are worse for the majorioty classes and because the test set mainly contains the majorty classes the overall accuracy is lower. 

# Conclusions and Recommendations

- Overall, based on the validation accuracy the baseline RF creates the best generalized model. 

- The validation accuracy during training of the cost-sensitive RNN has potential to be higher by decreasing the number of epochs. It is recommended to perform a more broad grid-search.

- The number of classes to predict for test is low and belong to the majority classes present in train. Due to  this imbalance the model performs well on the test set. There is no perfect solution for having only 13 labels in the test set.

- Look more into Decision Trees, as can be more robust with imbalanced datasets. Use One-vs-Rest strategy.      Create one RF model for each class. Give a sample as input to each model and pick the model with the highest score. For each minority class train each model on all samples that contain the class label and randomly select the same amount of data from the rest of the data. For each majority class randomly select samples from the class with the same amount of all other class instances.

- Make sure that both train and test sets are chosen randomly while having approximately the same representation ratios of examples from each feature. This can be done using resampling strategies for imbalanced datasets [2].  

- Try out different regularization techniques to reduce the gap between train and validation accuracies for the RNN. 

- Learning from imbalanced sets of data is still an open problem.

- More data :)

# References
[1] https://arxiv.org/pdf/1503.04069.pdf

[2] http://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf