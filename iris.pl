% Load the Iris dataset
load_iris :-
    csv_read_file('/Users/joyzhao/Desktop/iris.csv', [row(sepal_length,sepal_width,petal_length,petal_width,species)|Data], [functor(row)]),
    assert_iris(Data).

% Assert the Iris dataset as facts
assert_iris([]).
assert_iris([row(SepalLength, SepalWidth, PetalLength, PetalWidth, Species)|Rest]) :-
    assert(iris(SepalLength, SepalWidth, PetalLength, PetalWidth, Species)),
    assert_iris(Rest).

% Compute the distance between two Iris samples
distance(Sample1, Sample2, Distance) :-
    iris(X1, Y1, Z1, W1, Sample1),
    iris(X2, Y2, Z2, W2, Sample2),
    Distance is sqrt((X1-X2)^2 + (Y1-Y2)^2 + (Z1-Z2)^2 + (W1-W2)^2).

% Find the k nearest neighbors of a given Iris sample
k_nearest_neighbors(Sample, K, Neighbors) :-
    findall(Distance-Species, (iris(_, _, _, _, Species), distance(Sample, Species, Distance)), Distances),
    keysort(Distances, Sorted),
    take(K, Sorted, Neighbors),
    write('k_nearest_neighbors: '), write(Neighbors), nl.

% Take the first N elements of a list
take(0, _, []).
take(N, [X|Xs], [X|Ys]) :- N > 0, M is N-1, take(M, Xs, Ys).

% Classify an Iris sample using the k-NN algorithm
classify(Sample, K, Class) :-
    k_nearest_neighbors(Sample, K, Neighbors),
    count_neighbors(Neighbors, Counts),
    keysort(Counts, Sorted),
    reverse(Sorted, [Class-_|_]),
    format('Predicted class: ~w~n', [Class]).

% Count the occurrences of each species in a list of neighbors
count_neighbors([], []).
count_neighbors([_-Species|Neighbors], [(Count-Species)|Counts]) :-
    count_species(Species, Neighbors, Count),
    count_neighbors(Neighbors, Counts),
    write('count_neighbors: '), write(Counts), nl.

% Count the occurrences of a species in a list of neighbors
count_species(Species, Neighbors, Count) :-
    include(=(_-Species), Neighbors, Matches),
    length(Matches, Count).

