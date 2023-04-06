% try:
% ?- knn(3, 4.0, 3.4, 1.5, 0.2, Class, Neighbors).
% Define the distance between two instances, euclidian distance
% X = Sepal Length 
% Y = Sepal Width 
% Z = Petal Length 
% W = Petal Width 
% Class = Species

distance(X1,Y1,Z1,W1,X2,Y2,Z2,W2,Dist) :-
    Dist is sqrt((X1-X2)**2 + (Y1-Y2)**2 + (Z1-Z2)**2 + (W1-W2)**2).

% Define the KNN algorithm
knn(K,X,Y,Z,W,Class,Neighbors) :-
    findall(Distance-Class1, (data(X1,Y1,Z1,W1,Class1), distance(X,Y,Z,W,X1,Y1,Z1,W1,Distance)), Distances),
    sort(Distances, SortedDistances),
    take(K, SortedDistances, Nearest),
    get_neighbors(Nearest, Neighbors),
    get_classes(Neighbors, Classes),
    classify(Classes, Class).

% Helper function to take the first K elements of a list
take(0, _, []) :- !.
take(_, [], []) :- !.
take(K, [H|T], [H|R]) :- K1 is K-1, take(K1, T, R).

% Helper function to extract the neighbors with their distances
get_neighbors([], []) :- !.
get_neighbors([Distance-Class1|T], [FormattedDistance-Class1|R]) :-
    format(atom(FormattedDistance), '~3f', [Distance]), % format distance to 3 decimal places
    get_neighbors(T, R).

% Helper function to extract the class labels from the neighbors
get_classes([], []) :- !.
get_classes([_-C|T], [C|R]) :- get_classes(T, R).

% Define the classify function to return the most frequent class in the neighbors
classify(Neighbors, Class) :-
    sort(Neighbors, SortedNeighbors),
    reverse(SortedNeighbors, ReverseNeighbors),
    get_most_frequent(ReverseNeighbors, Class).

% Helper function to get the most frequent element in a list
get_most_frequent([H|T], H) :- count_occurrences(H, [H|T], Count), \+ (member(X, T), count_occurrences(X, [H|T], Count1), Count1 > Count).
get_most_frequent([H|T], R) :- get_most_frequent(T, R), count_occurrences(R, [H|T], CountR), count_occurrences(H, [H|T], CountH), CountR > CountH.

% Helper function to count the occurrences of an element in a list
count_occurrences(_, [], 0).
count_occurrences(X, [X|T], N) :- count_occurrences(X, T, N1), N is N1 + 1.
count_occurrences(X, [Y|T], N) :- X \= Y, count_occurrences(X, T, N). 


% Begin Cross Validation code 
% Try ?- cross_validation(3, 5, Accuracy).
% Steps:
% k_fold_cross_validation perfoms 5-fold cross validation on the data 
    % 1. shuffles dataset 
    % 2. split into 5 folds
    % 3. for each fold uses the other folds as training data and current fold as test fold 
    % 4. trains using knn on training set and evaluates of test set
    % 5. saves accuracy score over all folds and returns the average. 

cross_validation(K, Folds, Accuracy) :-
    findall((X,Y,Z,W,Class), data(X,Y,Z,W,Class), Instances),
    random_permutation(Instances, ShuffledInstances),
    % write('Shuffled Instances: '), writeln(ShuffledInstances), % uncomment to view ShuffledInstances 
    length(ShuffledInstances, NumInstances),
    num_folds(NumInstances, Folds, InstancesPerFold, Remainder),
    split_data(ShuffledInstances, Folds, InstancesPerFold, Remainder, SplitData),
    % write('Split Data: '), writeln(SplitData), %uncomment to view SplitData

    evaluate_folds(SplitData, K, AccuracyList),
    write('Accuracy of folds: '), writeln(AccuracyList),
    standard_deviation(AccuracyList, Std),
    write('Standard deviation: '), writeln(Std),
    sum_list(AccuracyList, TotalAccuracy),
    Accuracy is TotalAccuracy / Folds.


% Helper function to split the data into n folds
% split_data is helper function that splits dataset into n folds
    % takes dataset, number of folds to use, number of instances per fold, 
    % remainder (num instances that couldn't be evenly distributed among the folds)
    % and list of n empty lists representing the folds. first splits dataset into n parts, 
    % each with same number of instances, distributes any remaining evenly among first few folds 
    % retuns a list of n folds

split_data([], _, _, _, []).
split_data(Instances, Folds, InstancesPerFold, Remainder, [Fold|FoldsList]) :-
    length(Fold, InstancesPerFold),
    append(Fold, Rest, Instances),
    ( Remainder > 0 ->
        length(RestFold, InstancesPerFold+1),
        append(RestFold, NewRest, Rest),
        Remainder1 is Remainder - 1,
        split_data(NewRest, Folds-1, InstancesPerFold, Remainder1, FoldsList)
    ;
        split_data(Rest, Folds-1, InstancesPerFold, Remainder, FoldsList)
    ).

% Helper predicate to calculate the number of instances per fold and remainder
num_folds(NumInstances, NumFolds, InstancesPerFold, Remainder) :-
    InstancesPerFold is NumInstances // NumFolds,
    Remainder is NumInstances mod NumFolds.


% Helper function to evaluate the accuracy score of each fold

% 3-fold cross-validation
% evaluate_folds([], _, []).
% evaluate_folds([Fold1, Fold2, Fold3], K, Accuracies) :-
%     evaluate_fold([Fold2, Fold3], Fold1, K, Acc1),
%     evaluate_fold([Fold3, Fold1], Fold2, K, Acc2),
%     evaluate_fold([Fold1, Fold2], Fold3, K, Acc3),
%     Accuracies = [Acc1, Acc2, Acc3].

% 5-fold cross-validation
evaluate_folds([], _, []).
evaluate_folds([Fold1, Fold2, Fold3, Fold4, Fold5], K, Accuracies) :-
    evaluate_fold([Fold2, Fold3, Fold4, Fold5], Fold1, K, Acc1),
    evaluate_fold([Fold3, Fold4, Fold5, Fold1], Fold2, K, Acc2),
    evaluate_fold([Fold4, Fold5, Fold1, Fold2], Fold3, K, Acc3),
    evaluate_fold([Fold5, Fold1, Fold2, Fold3], Fold4, K, Acc4),
    evaluate_fold([Fold1, Fold2, Fold3, Fold4], Fold5, K, Acc5),
    Accuracies = [Acc1, Acc2, Acc3, Acc4, Acc5].

evaluate_fold(TrainingFolds, TestFold, K, Accuracy) :-
    findall((X,Y,Z,W,Class), (member((X,Y,Z,W,Class), TestFold), 
    \+ knn_train(K,X,Y,Z,W,Class,_, TrainingFolds)), Incorrect),
    length(TestFold, NumTestInstances),
    length(Incorrect, NumIncorrect),
    Accuracy is 1 - (NumIncorrect / NumTestInstances).

% same knn model but only uses training data when finding neighbors
knn_train(K, X, Y, Z, W, Class, Neighbors, TrainingFolds) :-
    findall(Distance-Class1, (
        member(TrainingFold, TrainingFolds),
        member((X1, Y1, Z1, W1, Class1), TrainingFold),
        distance(X, Y, Z, W, X1, Y1, Z1, W1, Distance)
    ), Distances),
    sort(Distances, SortedDistances),
    take(K, SortedDistances, Nearest),
    get_neighbors(Nearest, Neighbors),
    get_classes(Neighbors, Classes),
    classify(Classes, Class).

% Define a predicate to calculate the mean of a list of numbers
mean(List, Mean) :-
    sum_list(List, Sum),
    length(List, Length),
    Mean is Sum / Length.

% Define a predicate to calculate the sum of the squares of the differences between each number and the mean
sum_squares_differences(List, Mean, SumSquares) :-
    maplist({Mean}/[X, Y]>>(Y is (X - Mean) ** 2), List, SquaredDifferences),
    sum_list(SquaredDifferences, SumSquares).

% Define a predicate to calculate the standard deviation
standard_deviation(List, StandardDeviation) :-
    mean(List, Mean),
    length(List, Length),
    sum_squares_differences(List, Mean, SumSquares),
    Variance is SumSquares / (Length - 1),
    StandardDeviation is sqrt(Variance).

% Iris dataset
data(5.1,3.5,1.4,0.2,'Iris-setosa').
data(4.9,3.0,1.4,0.2,'Iris-setosa').
data(4.7,3.2,1.3,0.2,'Iris-setosa').
data(4.6,3.1,1.5,0.2,'Iris-setosa').
data(5.0,3.6,1.4,0.2,'Iris-setosa').
data(5.4,3.9,1.7,0.4,'Iris-setosa').
data(4.6,3.4,1.4,0.3,'Iris-setosa').
data(5.0,3.4,1.5,0.2,'Iris-setosa').
data(4.4,2.9,1.4,0.2,'Iris-setosa').
data(4.9,3.1,1.5,0.1,'Iris-setosa').
data(5.4,3.7,1.5,0.2,'Iris-setosa').
data(4.8,3.4,1.6,0.2,'Iris-setosa').
data(4.8,3.0,1.4,0.1,'Iris-setosa').
data(4.3,3.0,1.1,0.1,'Iris-setosa').
data(5.8,4.0,1.2,0.2,'Iris-setosa').
data(5.7,4.4,1.5,0.4,'Iris-setosa').
data(5.4,3.9,1.3,0.4,'Iris-setosa').
data(5.1,3.5,1.4,0.3,'Iris-setosa').
data(5.7,3.8,1.7,0.3,'Iris-setosa').
data(5.1,3.8,1.5,0.3,'Iris-setosa').
data(5.4,3.4,1.7,0.2,'Iris-setosa').
data(5.1,3.7,1.5,0.4,'Iris-setosa').
data(4.6,3.6,1.0,0.2,'Iris-setosa').
data(5.1,3.3,1.7,0.5,'Iris-setosa').
data(4.8,3.4,1.9,0.2,'Iris-setosa').
data(5.0,3.0,1.6,0.2,'Iris-setosa').
data(5.0,3.4,1.6,0.4,'Iris-setosa').
data(5.2,3.5,1.5,0.2,'Iris-setosa').
data(5.2,3.4,1.4,0.2,'Iris-setosa').
data(4.7,3.2,1.6,0.2,'Iris-setosa').
data(4.8,3.1,1.6,0.2,'Iris-setosa').
data(5.4,3.4,1.5,0.4,'Iris-setosa').
data(5.2,4.1,1.5,0.1,'Iris-setosa').
data(5.5,4.2,1.4,0.2,'Iris-setosa').
data(4.9,3.1,1.5,0.1,'Iris-setosa').
data(5.0,3.2,1.2,0.2,'Iris-setosa').
data(5.5,3.5,1.3,0.2,'Iris-setosa').
data(4.9,3.1,1.5,0.1,'Iris-setosa').
data(4.4,3.0,1.3,0.2,'Iris-setosa').
data(5.1,3.4,1.5,0.2,'Iris-setosa').
data(5.0,3.5,1.3,0.3,'Iris-setosa').
data(4.5,2.3,1.3,0.3,'Iris-setosa').
data(4.4,3.2,1.3,0.2,'Iris-setosa').
data(5.0,3.5,1.6,0.6,'Iris-setosa').
data(5.1,3.8,1.9,0.4,'Iris-setosa').
data(4.8,3.0,1.4,0.3,'Iris-setosa').
data(5.1,3.8,1.6,0.2,'Iris-setosa').
data(4.6,3.2,1.4,0.2,'Iris-setosa').
data(5.3,3.7,1.5,0.2,'Iris-setosa').
data(5.0,3.3,1.4,0.2,'Iris-setosa').
data(7.0,3.2,4.7,1.4,'Iris-versicolor').
data(6.4,3.2,4.5,1.5,'Iris-versicolor').
data(6.9,3.1,4.9,1.5,'Iris-versicolor').
data(5.5,2.3,4.0,1.3,'Iris-versicolor').
data(6.5,2.8,4.6,1.5,'Iris-versicolor').
data(5.7,2.8,4.5,1.3,'Iris-versicolor').
data(6.3,3.3,4.7,1.6,'Iris-versicolor').
data(4.9,2.4,3.3,1.0,'Iris-versicolor').
data(6.6,2.9,4.6,1.3,'Iris-versicolor').
data(5.2,2.7,3.9,1.4,'Iris-versicolor').
data(5.0,2.0,3.5,1.0,'Iris-versicolor').
data(5.9,3.0,4.2,1.5,'Iris-versicolor').
data(6.0,2.2,4.0,1.0,'Iris-versicolor').
data(6.1,2.9,4.7,1.4,'Iris-versicolor').
data(5.6,2.9,3.6,1.3,'Iris-versicolor').
data(6.7,3.1,4.4,1.4,'Iris-versicolor').
data(5.6,3.0,4.5,1.5,'Iris-versicolor').
data(5.8,2.7,4.1,1.0,'Iris-versicolor').
data(6.2,2.2,4.5,1.5,'Iris-versicolor').
data(5.6,2.5,3.9,1.1,'Iris-versicolor').
data(5.9,3.2,4.8,1.8,'Iris-versicolor').
data(6.1,2.8,4.0,1.3,'Iris-versicolor').
data(6.3,2.5,4.9,1.5,'Iris-versicolor').
data(6.1,2.8,4.7,1.2,'Iris-versicolor').
data(6.4,2.9,4.3,1.3,'Iris-versicolor').
data(6.6,3.0,4.4,1.4,'Iris-versicolor').
data(6.8,2.8,4.8,1.4,'Iris-versicolor').
data(6.7,3.0,5.0,1.7,'Iris-versicolor').
data(6.0,2.9,4.5,1.5,'Iris-versicolor').
data(5.7,2.6,3.5,1.0,'Iris-versicolor').
data(5.5,2.4,3.8,1.1,'Iris-versicolor').
data(5.5,2.4,3.7,1.0,'Iris-versicolor').
data(5.8,2.7,3.9,1.2,'Iris-versicolor').
data(6.0,2.7,5.1,1.6,'Iris-versicolor').
data(5.4,3.0,4.5,1.5,'Iris-versicolor').
data(6.0,3.4,4.5,1.6,'Iris-versicolor').
data(6.7,3.1,4.7,1.5,'Iris-versicolor').
data(6.3,2.3,4.4,1.3,'Iris-versicolor').
data(5.6,3.0,4.1,1.3,'Iris-versicolor').
data(5.5,2.5,4.0,1.3,'Iris-versicolor').
data(5.5,2.6,4.4,1.2,'Iris-versicolor').
data(6.1,3.0,4.6,1.4,'Iris-versicolor').
data(5.8,2.6,4.0,1.2,'Iris-versicolor').
data(5.0,2.3,3.3,1.0,'Iris-versicolor').
data(5.6,2.7,4.2,1.3,'Iris-versicolor').
data(5.7,3.0,4.2,1.2,'Iris-versicolor').
data(5.7,2.9,4.2,1.3,'Iris-versicolor').
data(6.2,2.9,4.3,1.3,'Iris-versicolor').
data(5.1,2.5,3.0,1.1,'Iris-versicolor').
data(5.7,2.8,4.1,1.3,'Iris-versicolor').
data(6.3,3.3,6.0,2.5,'Iris-virginica').
data(5.8,2.7,5.1,1.9,'Iris-virginica').
data(7.1,3.0,5.9,2.1,'Iris-virginica').
data(6.3,2.9,5.6,1.8,'Iris-virginica').
data(6.5,3.0,5.8,2.2,'Iris-virginica').
data(7.6,3.0,6.6,2.1,'Iris-virginica').
data(4.9,2.5,4.5,1.7,'Iris-virginica').
data(7.3,2.9,6.3,1.8,'Iris-virginica').
data(6.7,2.5,5.8,1.8,'Iris-virginica').
data(7.2,3.6,6.1,2.5,'Iris-virginica').
data(6.5,3.2,5.1,2.0,'Iris-virginica').
data(6.4,2.7,5.3,1.9,'Iris-virginica').
data(6.8,3.0,5.5,2.1,'Iris-virginica').
data(5.7,2.5,5.0,2.0,'Iris-virginica').
data(5.8,2.8,5.1,2.4,'Iris-virginica').
data(6.4,3.2,5.3,2.3,'Iris-virginica').
data(6.5,3.0,5.5,1.8,'Iris-virginica').
data(7.7,3.8,6.7,2.2,'Iris-virginica').
data(7.7,2.6,6.9,2.3,'Iris-virginica').
data(6.0,2.2,5.0,1.5,'Iris-virginica').
data(6.9,3.2,5.7,2.3,'Iris-virginica').
data(5.6,2.8,4.9,2.0,'Iris-virginica').
data(7.7,2.8,6.7,2.0,'Iris-virginica').
data(6.3,2.7,4.9,1.8,'Iris-virginica').
data(6.7,3.3,5.7,2.1,'Iris-virginica').
data(7.2,3.2,6.0,1.8,'Iris-virginica').
data(6.2,2.8,4.8,1.8,'Iris-virginica').
data(6.1,3.0,4.9,1.8,'Iris-virginica').
data(6.4,2.8,5.6,2.1,'Iris-virginica').
data(7.2,3.0,5.8,1.6,'Iris-virginica').
data(7.4,2.8,6.1,1.9,'Iris-virginica').
data(7.9,3.8,6.4,2.0,'Iris-virginica').
data(6.4,2.8,5.6,2.2,'Iris-virginica').
data(6.3,2.8,5.1,1.5,'Iris-virginica').
data(6.1,2.6,5.6,1.4,'Iris-virginica').
data(7.7,3.0,6.1,2.3,'Iris-virginica').
data(6.3,3.4,5.6,2.4,'Iris-virginica').
data(6.4,3.1,5.5,1.8,'Iris-virginica').
data(6.0,3.0,4.8,1.8,'Iris-virginica').
data(6.9,3.1,5.4,2.1,'Iris-virginica').
data(6.7,3.1,5.6,2.4,'Iris-virginica').
data(6.9,3.1,5.1,2.3,'Iris-virginica').
data(5.8,2.7,5.1,1.9,'Iris-virginica').
data(6.8,3.2,5.9,2.3,'Iris-virginica').
data(6.7,3.3,5.7,2.5,'Iris-virginica').
data(6.7,3.0,5.2,2.3,'Iris-virginica').
data(6.3,2.5,5.0,1.9,'Iris-virginica').
data(6.5,3.0,5.2,2.0,'Iris-virginica').
data(6.2,3.4,5.4,2.3,'Iris-virginica').
data(5.9,3.0,5.1,1.8,'Iris-virginica').


% evaluate_folds(Folds, K, Accuracies) :-
%     length(Folds, NumFolds),
%     evaluate_folds_helper(Folds, NumFolds, K, [], Accuracies, NumFolds).

% evaluate_folds_helper(_, 0, _, Accuracies, Accuracies, _).
% evaluate_folds_helper(Folds, N, K, Acc, Accuracies, NumFolds) :-
%     N > 0,
%     evaluate_fold(Folds, Fold, K, Acc1),
%     N1 is N - 1,
%     append(Acc, [Acc1], Acc2),
%     rotate_list(Folds, RotatedFolds),
%     evaluate_folds_helper(RotatedFolds, N1, K, Acc2, Accuracies, NumFolds).

% rotate_list([H | T], R) :-
%     append(T, [H], R).