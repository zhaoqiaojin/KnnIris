% % Define the distance between two instances
% distance(X1,Y1,Z1,W1,X2,Y2,Z2,W2,Dist) :-
%     Dist is sqrt((X1-X2)**2 + (Y1-Y2)**2 + (Z1-Z2)**2 + (W1-W2)**2).

% % Define the KNN algorithm
% knn(K,X,Y,Z,W,Class,Neighbors) :-
%     findall(Distance-Class1, (data(X1,Y1,Z1,W1,Class1), distance(X,Y,Z,W,X1,Y1,Z1,W1,Distance)), Distances),
%     sort(Distances, SortedDistances),
%     take(K, SortedDistances, Nearest),
%     get_neighbors(Nearest, Neighbors),
%     get_classes(Neighbors, Classes),
%     classify(Classes, Class).

% % Helper function to take the first K elements of a list
% take(0, _, []) :- !.
% take(_, [], []) :- !.
% take(K, [H|T], [H|R]) :- K1 is K-1, take(K1, T, R).

% % Helper function to extract the neighbors with their distances
% get_neighbors([], []) :- !.
% get_neighbors([Distance-Class1|T], [FormattedDistance-Class1|R]) :-
%     format(atom(FormattedDistance), '~3f', [Distance]), % format distance to 3 decimal places
%     get_neighbors(T, R).

% % Helper function to extract the class labels from the neighbors
% get_classes([], []) :- !.
% get_classes([_-C|T], [C|R]) :- get_classes(T, R).

% % Define the classify function to return the most frequent class in the neighbors
% classify(Neighbors, Class) :-
%     sort(Neighbors, SortedNeighbors),
%     reverse(SortedNeighbors, ReverseNeighbors),
%     get_most_frequent(ReverseNeighbors, Class).

% % Helper function to get the most frequent element in a list
% get_most_frequent([H|T], H) :- count_occurrences(H, [H|T], Count), \+ (member(X, T), count_occurrences(X, [H|T], Count1), Count1 > Count).
% get_most_frequent([H|T], R) :- get_most_frequent(T, R), count_occurrences(R, [H|T], CountR), count_occurrences(H, [H|T], CountH), CountR > CountH.

% % Helper function to count the occurrences of an element in a list
% count_occurrences(_, [], 0).
% count_occurrences(X, [X|T], N) :- count_occurrences(X, T, N1), N is N1 + 1.
% count_occurrences(X, [Y|T], N) :- X \= Y, count_occurrences(X, T, N). 


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

% implementation fo PCA on the iris dataset 

:- use_module(library(csv)).
:- use_module(library(clpfd)).

% Define the dataset as facts
data(5.1,3.5,1.4,0.2,'Iris-setosa').
data(4.9,3.0,1.4,0.2,'Iris-setosa').
data(4.7,3.2,1.3,0.2,'Iris-setosa').
data(4.6,3.1,1.5,0.2,'Iris-setosa').
% ... more data points ...

% Define the number of attributes
num_attributes(4).

% Define the number of principal components
num_components(2).

% Define the output CSV file
output_file('iris_pca.csv').

% Entry point for the PCA algorithm
run_pca :-
    % Step 1: Collect the data points and store them in a list of lists
    findall(Data, (data(A,B,C,D,Species), Data=[A,B,C,D]), DataList),

    % Step 2: Center the data around zero
    transpose(DataList, Transposed),
    maplist(mean, Transposed, MeanList),
    maplist(subtract(MeanList), DataList, CenteredList),

    % Step 3: Calculate the covariance matrix
    covariance_matrix(CenteredList, Covariance),

    % Step 4: Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigen(Covariance, Eigenvalues, Eigenvectors),

    % Step 5: Sort the eigenvectors in descending order based on their corresponding eigenvalues
    sort_eigenvectors(Eigenvalues, Eigenvectors, SortedEigenvectors),

    % Step 6: Take the top two eigenvectors and form a matrix from them
    take(num_components, SortedEigenvectors, Components),
    transpose(Components, ComponentsMatrix),

    % Step 7: Multiply the centered data by the eigenvector matrix to obtain the two principal components
    matrix_multiply(CenteredList, ComponentsMatrix, PCA),

    % Step 8: Add the species column to the principal components matrix and save it to a CSV file
    add_species_column(PCA, SpeciesList, PCASpecies),
    output_file(OutputFile),
    csv_write_file(OutputFile, PCASpecies).

% Helper predicates

% Calculate the mean of a list of numbers
mean(List, Mean) :-
    length(List, N),
    sumlist(List, Sum),
    Mean is Sum / N.c

% Calculate the dot product of two lists
dot_product(List1, List2, DotProduct) :-
    maplist(times, List1, List2, Products),
    sum(Products, DotProduct).

% Calculate the covariance matrix of a list of centered data points
covariance_matrix(CenteredData, Covariance) :-
    length(CenteredData, N),
    transpose(CenteredData, Transposed),
    maplist(covariance_helper(CenteredData), Transposed, CovarianceRows),
    matrix(CovarianceRows, Covariance).

% Helper predicate for covariance matrix calculation
covariance_helper(CenteredData, Column, CovarianceRow) :-
    maplist(times, CenteredData, Column, Products),
    sum(Products, CovarianceSum),
    CovarianceValue is CovarianceSum / (N-1),
    append(CovarianceRow, [CovarianceValue], CovarianceRow).


% Multiply two matrices
matrix_multiply(A, B, Result) :-
    transpose(B, Transposed),
    maplist(matrix_multiply_row(A), Transposed, ResultRows),
    transpose(ResultRows, Result).

% Multiply a matrix row by a matrix
matrix_multiply_row(Matrix, Row, Result) :-
    maplist(dot_product(Row), Matrix, Result).

% Divide a number by a constant
divide_by(Const, Num, Result) :-
    Result #= Num / Const.

% Calculate the eigenvalues and eigenvectors of a matrix
eigen(Matrix, Eigenvalues, Eigenvectors) :-
    eigen(Matrix, Eigenvectors, Eigenvalues, []).

% Sort the eigenvectors based on their corresponding eigenvalues
sort_eigenvectors(Eigenvalues, Eigenvectors, SortedEigenvectors) :-
    pairs_keys_values(Pairs, Eigenvalues, Eigenvectors),
    keysort(Pairs, SortedPairs),
    pairs_values(SortedPairs, SortedEigenvectors).

% Take the first N elements from a list
take(N, List, Result) :-
    length(Result, N),
    append(Result, _, List).

% Add a species column to a matrix
add_species_column(Matrix, SpeciesList, Result) :-
    maplist(add_species_column_helper, Matrix, SpeciesList, Result).

% Helper predicate for adding a species column to a matrix
add_species_column_helper(Row, Species, [Row, Species]).


% to run, try: ?- run_pca.