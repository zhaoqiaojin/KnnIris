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


% implement PCA on the iris dataset:

% Principal Component Analysis
pca(Features, Components) :-
    % Calculate the covariance matrix
    cov(Features, Covariance),
    % Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigen(Covariance, Eigenvalues, Eigenvectors),
    % Sort the eigenvalues and eigenvectors in descending order of eigenvalues
    sort_eigen(Eigenvalues, Eigenvectors, SortedEigenvalues, SortedEigenvectors),
    % Select the top k eigenvectors to use as principal components
    select_k_components(SortedEigenvectors, Components).

% Calculate the covariance matrix of the features
cov(Features, Covariance) :-
    length(Features, NumFeatures),
    numlist(1, NumFeatures, Indices),
    maplist(mean(Features), Indices, Means),
    maplist(deviation(Means), Features, Deviations),
    transpose(Deviations, DeviationsT),
    matrix_multiply(DeviationsT, Deviations, Covariance),
    scalar_multiply(1/(NumFeatures-1), Covariance, Covariance).

% Calculate the mean of a feature at a given index
mean(Features, Index, Mean) :-
    maplist(nth1(Index), Features, Values),
    sum_list(Values, Sum),
    length(Features, NumFeatures),
    Mean is Sum/NumFeatures.

% Calculate the deviation of a feature from its mean
deviation(Mean, Feature, Deviation) :-
    maplist(subtract(Mean), Feature, Deviation).

% Calculate the eigenvalues and eigenvectors of a matrix
eigen(Matrix, Eigenvalues, Eigenvectors) :-
    svd(Matrix, _, EigenvectorsT, SingularValues),
    transpose(EigenvectorsT, Eigenvectors),
    maplist(square, SingularValues, SquaredSingularValues),
    maplist(divide_by_sum(SquaredSingularValues), SquaredSingularValues, Eigenvalues).

% Square a number
square(Number, Result) :-
    Result is Number*Number.

% Divide a number by the sum of a list of numbers
divide_by_sum(List, Number, Result) :-
    sum_list(List, Sum),
    Result is Number/Sum.

% Sort the eigenvalues and eigenvectors in descending order of eigenvalues
sort_eigen(Eigenvalues, Eigenvectors, SortedEigenvalues, SortedEigenvectors) :-
    sort_eigen_descending(Eigenvalues, SortedEigenvalues, Indices),
    maplist(nth1(Indices), Eigenvectors, SortedEigenvectors).

% Sort a list of numbers in descending order and return the indices of the original order
sort_eigen_descending(List, SortedList, Indices) :-
    msort(List, SortedList),
    reverse_indices(List, SortedList, Indices).

% Get the indices of the elements in the first list in reverse order
reverse_indices(List, SortedList, Indices) :-
    maplist(reverse_index(List, SortedList), SortedList, Indices).

% Get the index of an element in the first list in reverse order
reverse_index(List, Element, Index) :-
    nth0(Index, List, Element).

% Select the top k eigenvectors to use as principal components
select_k_components(Eigenvectors, Components) :-
    length(Eigenvectors, NumComponents),
    select_k_components(NumComponents, Eigenvectors, Components).

select_k_components(K, Eigenvectors, Components) :-
    length(Components, K),
    select_k_components(K, Eigenvectors, Components, []).

% Select the top k eigenvectors to use as principal components
select_k_components(K, Eigenvectors, Components) :-
    length(Components, K),
    select_k_components(K, Eigenvectors, Components, []).

select_k_components(0, _, [], _).
select_k_components(K, Eigenvectors, [Component|Components], UsedIndices) :-
    length(Eigenvectors, NumEigenvectors),
    random_between(0, NumEigenvectors-1, Index),
    \+ memberchk(Index, UsedIndices),
    nth0(Index, Eigenvectors, Component),
    K1 is K-1,
    select_k_components(K1, Eigenvectors, Components, [Index|UsedIndices]).

% Multiply a matrix by a scalar
scalar_multiply(Scalar, Matrix, Result) :-
    maplist(scalar_multiply_row(Scalar), Matrix, Result).

% Multiply a row by a scalar
scalar_multiply_row(Scalar, Row, Result) :-
    maplist(multiply(Scalar), Row, Result).

% Multiply two numbers
multiply(A, B, C) :-
    C is A*B.

% Multiply two matrices
matrix_multiply(A, B, C) :-
    transpose(B, BT),
    maplist(dot_product(BT), A, C).

% Calculate the dot product of two vectors
dot_product(A, B, C) :-
    maplist(multiply, A, B, Products),
    sum_list(Products, C).

% Project the data onto the principal components
project(Features, Components, ProjectedFeatures) :-
    matrix_multiply(Features, Components, ProjectedFeatures).

% Apply PCA to the features
features(F), pca(F, Components).

% Project the features onto the principal components
project(F, Components, ProjectedF), writeln(ProjectedF).



% Export the projected features to a CSV file
export_to_csv(File, Data) :-
    atomic_list_concat(Data, '\n', CSVData),
    open(File, write, Stream),
    write(Stream, CSVData),
    close(Stream).

% Apply PCA to the features
features(F), pca(F, Components), project(F, Components, ProjectedF),

% Export the projected features to a CSV file
export_to_csv('iris_pca.csv', ProjectedF).