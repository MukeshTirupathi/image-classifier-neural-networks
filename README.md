# image-classifier-neural-networks
 Neural Networks for Classification  
In this notebook we are going to explore the use of Neural Networks for image classification. We are going to use a dataset of small images of clothes and accessories, the Fashion MNIST. You can find more information regarding the dataset here: https://pravarmahajan.github.io/fashion/  Each instance in the dataset consist of an image, in a format similar to the digit images you have seen in the previous homework, and a label. 
The labels correspond to the type of clothing, as follows:
| Label | Description | 
| 0 | T-shirt/top | 
| 1 | Trouser | 
| 2 | Pullover | 
| 3 | Dress | 
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

we use
1.from sklearn.neural_network import MLPClassifier
2.from sklearn.model_selection import GridSearchCV

Now use a (feed-forward) Neural Network for prediction. Use the multi-layer perceptron (MLP) classifier MLPClassifier(...) in scikit-learn, with the following parameters: max_iter=300, alpha=1e-4, solver='sgd', tol=1e-4, learning_rate_init=.1, random_state=ID (this last parameter ensures the run is the same even if you run it more than once). The alpha parameter is the regularization parameter for L2 regularization that is used by the MLP in sklearn.

Then, using the default activation function, pick four or five architectures to consider, with different numbers of hidden layers and different sizes. It is not necessary to create huge neural networks, you can limit to 3 layers and, for each layer, its maximum size can be of 100. You can evaluate the architectures you chose using the GridSearchCV with a 5-fold cross-validation, and use the results to pick the best architecture. The code below provides some architectures you can use, but you can choose other ones if you prefer.

result:
The above results show that 100 nuerons with single hidden layer gives the best performance than the 50 and 10 nuerons with single hidden layer.
the two hidden layers with 100 nuerons also gives best result compared to 50 and 10 nuerons with 2 hidden layer
Hence we can tell that the increase in nuerons gives best performance than increasing the no of hidden layers.(Even with more data samples=10000)

Pick another classifier among the ones we have seen previously (SVM or something else). Report the training and test error for such classifier with 10000 samples in the training set, if possible; if the classifier cannot run with so many data sample reduce the number of samples.

*Note*: if there are parameters to be optimized use cross-validation. If you choose SVM, you can decide if you want to use a single kernel or use the best among many; in the latter case, you need to pick the best kernel using cross-validation (using the functions available in sklearn).

The classifier used below is SVM with RBF and the parameters are 'C': [10], 'gamma': [0.01] which we found in the SVM homework that the RBF with the parameteres 'C': [10], 'gamma': [0.01] is showing better results than linear and poly2.

And i decided to use SVM because unstructured and semi-structured data like text and images while logistic regression works with already identified independent variables.

Clustering with K-means

Clustering is a useful technique for *unsupervised* learning. We are now going to cluster 2000 images in the fashion MNIST dataset, and try to understand if the clusters we obtain correspond to the true labels.

Choice of k with silhoutte coefficient
In many real applications it is unclear what is the correct value of $k$ to use. In practice one tries different values of $k$ and then uses some external score to choose a value of $k$. One such score is the silhoutte coefficient, that can be computed with metrics.silhouette_score(). See the definition of the silhoutte coefficient in the userguide.
