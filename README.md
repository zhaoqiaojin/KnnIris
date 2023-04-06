# kNN Classification of the Iris Dataset

We created a kNN model on the iris dataset using all four features to predict the species (class) of the iris. The kNN model can take an arbitrary user defiend k-neighbors and compute the most common class of the neighbros as the predicted class. In the case that there is a tie between two classes for neighbor instances, the typing class with the smallest distance to the given prediction data will be returned. 

For our "extra" of this project, we created a cross-validation test on our kNN model by splitting the iris dataset into test set and training sets to compute the average accuracy score of the fold and also list out the accuracies of each fold as well as the standard deviation. 

## Overview

The [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) is a classic and widely-used benchmark dataset in the field of machine learning. It consists of 150 samples of iris flowers, with 50 samples from each of three species: setosa, versicolor, and virginica. For each sample, four features are measured: sepal length, sepal width, petal length and petal width. The Iris dataset is often used for classification tasks and is a good example of a small, well-understood dataset for testing machine learning algorithms.

Here are some example data from the Iris dataset. 

| Sepal Length | Sepal Width | Petal Length | Petal Width | Species |
|:------------:|:-----------:|:------------:|:-----------:|:-------:|
|     5.1      |     3.5     |     1.4      |     0.2     |  setosa |
|     4.9      |     3.0     |     1.4      |     0.2     |  setosa |
|     6.5      |     3.0     |     5.2      |     2.0     | virginica|
|     6.0      |     2.2     |     5.0      |     1.5     | virginica|
|     5.5      |     2.3     |     4.0      |     1.3     |versicolor|
|     5.7      |     2.8     |     4.1      |     1.3     |versicolor|


The data was loaded into Prolog directly as facts:
```prolog
% Iris dataset
data(5.1,3.5,1.4,0.2,'Iris-setosa').
data(4.9,3.0,1.4,0.2,'Iris-setosa').
```

## Usage

**kNN Classification**

Call to knn function takes K (k neighbors), the four features (X = Sepal Length), (Y = Sepal Width), (Z = Petla Length), (W = Petal Width), and outputs the predicted Class/Species and list of k-neighbors with distances rounded to 3 decimal places and well as neighbor's class. 

``` prolog 
knn(K,X,Y,Z,W,Class,Neighbors):-
```

```bash
?- knn(3, 4.0, 3.4, 1.5, 0.2, Class, Neighbors).
Class = 'Iris-setosa',
Neighbors = ['0.490'-'Iris-setosa', '0.600'-'Iris-setosa', '0.616'-'Iris-setosa'] 

```
also try: 
```bash
?- knn(1, 6.3, 2.3, 5.1, 1.5, Class, Neighbors).
?- knn(5, 5.6, 2.4, 4.1, 1.2, Class, Neighbors).

```

**Cross-validation Score**

Call to cross_validation takes K (k-neighbors), Folds (num folds) and outpput list of the accuracies of each fold, standard deviation of accuracies, and finally average accuracy across folds. 

``` prolog 
cross_validation(K, Folds, Accuracy) :-
```

```bash 
?- cross_validation(3, 5, Accuracy). 
Accuracy of folds: [0.9,1,0.9333333333333333,0.9666666666666667,0.9333333333333333]
Standard deviation: 0.03800584750330459
Accuracy = 0.9466666666666667 .
```


## Improvements:

* Make a visual representation of the iris dataset by plotting a scatterplot and color coding the classes 
    * Because there are 4 features, we can first perform PCA on the dataset and then visualize
* Due to time constraints we did not perform any preprocessing on the original dataset before implementing our kNN model. Although the features were generally on the same scale, it would have been better to perform preprocessing such as standard scalar on the four features before creating the kNN model. 

## Key Takeaways:

* Prolog is a powerful programming language that is well-suited for implementing some machine learning algorithms, although lacking in many built in functions such as the ones found in python's scikit-learn library (such as preprocessing of data or PCA) being able to code the kNN model directly and explicitly in prolog was very rewarding
* KNN is a simple yet effective machine learning algorithm that can be used for classification and regression tasks, our model generally performed at 96% cross-validation accuracy score for a neighbours value of 3.
* You can make predicates in Prolog that can be used to generate a list of solutions to a query like the findall predicate in our project. Another important takeaway is the use of Prolog's built-in predicates, such as findall, sort, and length, to perform operations on lists of data points.
* We also learned how to perform cross-validation to evaluate the performance of the model. It was interesting to compute cross-validation accuracy score from scratch in Prolog without relying on built-in libraries such as the scikit-learn library in Python. Cross-validation is a widely used technique for evaluating machine learning models, as it provides a more accurate estimate of the model's performance than a simple train-test split. To make sure our model was performing correctly, used the cross-validation scores for different k values and folds from our prolog project and compared to the ones produced by python scikit-learn library. They complemented each other with minor differences due to the ransom shuffling of data when performing the splits, meaning that our kNN model works as expected.

## Acknowledgements

Original inspiration for this project idea taken from this example SWISH program of [EM clustering of the Iris dataset](https://swish.swi-prolog.org/example/iris.swinb) 


## License

[MIT](https://choosealicense.com/licenses/mit/)
