
% Define the distance between two instances
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
% Steps:
% Divide the dataset into 5-folds, we will perform 5-fold cross validation since the dataset has 150 entries 
% 5 x 30 is a pretty good amount of data for each evaluation 
% For each fold i, use it as the test set and the remaining folds as the training set. 
% Then, train your KNN model on the training set and evaluate it on the test set. 
% Save the evaluation metric, such as accuracy or F1-score, for this fold.
% Repeat step 2 for all 5 folds, so that each fold is used as the test set once.
% Compute the average evaluation metric over all 5 folds. 
% This will give us the cross-validation score for our KNN model.
% call the cross validation function using ?-cross_validate(K, Score) where K is the desire K for our KNN

% Define the number of folds for cross-validation
:- dynamic num_folds/1.
num_folds(5).

% Define the cross-validation function
cross_validate(K, Score) :-
    % Get the number of folds
    num_folds(N),

    % Shuffle the dataset
    findall(data(X,Y,Z,W,Class), data(X,Y,Z,W,Class), Data),
    random_permutation(Data, ShuffledData),

    % Divide the dataset into N folds
    split_list(ShuffledData, N, Folds),

    % Evaluate the KNN model on each fold
    findall(FoldScore, (
        nth0(I, Folds, TestFold),
        exclude_nth(I, Folds, TrainFolds),
        flatten(TrainFolds, TrainData),
        maplist(predict(K, TrainData), TestFold, TestLabels),
        maplist(check_prediction, TestFold, TestLabels, FoldScore)
    ), FoldScores),

    % Compute the average score over all folds
    sum_list(FoldScores, TotalScore),
    length(FoldScores, NumFolds),
    Score is TotalScore / NumFolds.


% Define the predict function to predict the label of a test instance using KNN
predict(K, TrainData, Test, Label) :-
    Test = data(X,Y,Z,W,_),
    knn(K, X, Y, Z, W, Label, Neighbors),
    get_neighbors_classes(Neighbors, TrainData, Classes),
    classify(Classes, Label),
    format("Test instance: ~w, Predicted label: ~w~n", [Test, Label]),
    format("Neighbors: ~w~n", [Neighbors]),
    format("Classes: ~w~n", [Classes]).

% Helper function to get the classes of the neighbors
get_neighbors_classes([], _, []).
get_neighbors_classes([_-C|T], TrainData, [C|R]) :-
    member(data(_,_,_,_,C), TrainData),
    get_neighbors_classes(T, TrainData, R).
get_neighbors_classes([_-C|T], TrainData, R) :-
    \+ member(data(_,_,_,_,C), TrainData),
    get_neighbors_classes(T, TrainData, R).



% Helper function to check if the prediction is correct
check_prediction(Test, PredictedLabel, Score) :-
    Test = data(_,_,_,_,TrueLabel),
    (PredictedLabel = TrueLabel -> Score = 1 ; Score = 0).

% Helper function to exclude the nth element from a list
exclude_nth(N, List, Excluded) :-
    nth0(N, List, _, WithoutNth),
    append(WithoutNth, Excluded).

% Split a list into sublists of equal length
split_list(List, NumSublists, Sublists) :-
    length(List, NumElements),
    ChunkSize is ceil(NumElements / NumSublists),
    write('ChunkSize: '), write(ChunkSize), nl,
    split_list_helper(List, ChunkSize, Sublists),
    write('Sublists: '), write(Sublists), nl.

split_list_helper([], _, []).
split_list_helper(List, ChunkSize, [Head|Sublists]) :-
    length(Head, ChunkSize),
    append(Head, Tail, List),
    write('Head: '), write(Head), nl,
    split_list_helper(Tail, ChunkSize, Sublists).

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

