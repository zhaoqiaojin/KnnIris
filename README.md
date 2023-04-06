# kNN Classification of the Iris Dataset

Foobar is a Python library for dealing with word pluralization.

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

Call to knn function takes K (k neighbors), the four features X, Y, Z, W, and outputs the predicted Class and list of k-neighbors with distances rounded to 3 decimal places and well as neighbor's class. 

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

## Key Takeaways:

## Acknowledgements

Original inspiration for this project idea taken from this example SWISH program of [EM clustering of the Iris dataset](https://swish.swi-prolog.org/example/iris.swinb) 


## License

[MIT](https://choosealicense.com/licenses/mit/)