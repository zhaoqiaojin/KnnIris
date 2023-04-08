:- consult('iris.pl').
:- use_module(library(plunit)).

:- begin_tests(distance_test).

    test(distance_between_origin_and_origin) :-
        distance(0, 0, 0, 0, 0, 0, 0, 0, Dist),
        assertion(Dist =:= 0).

    test(distance_between_two_points) :-
        distance(1, 1, 1, 1, 2, 2, 2, 2, Dist),
        assertion(Dist =:= 2).

:- end_tests(distance_test).

:- begin_tests(take_test).

    test(take_0, Result == []) :-
        take(0, [1,2,3,4,5], Result).

    test(take_all, Result == [1,2,3,4,5]) :-
        take(5, [1,2,3,4,5], Result).

    test(take_some, Result == [1,2,3]) :-
        take(3, [1,2,3,4,5], Result).

    test(take_none, Result == []) :-
        take(3, [], Result).

:- end_tests(take_test).

:- begin_tests(mean_test).

    test(empty_list, Mean == 0) :-
        mean([], Mean).

    test(single_element_list, Mean == 5) :-
        mean([5], Mean).

    test(multiple_elements_list, Mean == 3) :-
        mean([1,2,3,4,5], Mean).

:- end_tests(mean_test).

:- begin_tests(standard_deviation_test).

    test(empty_list, StandardDeviation == 0) :-
        standard_deviation([], StandardDeviation).

    test(single_element_list, StandardDeviation == 0) :-
        standard_deviation([1], StandardDeviation).

    test(multiple_elements_list, StandardDeviation == 5.310367218940701) :-
        standard_deviation([3, 5, 5, 6, 7, 8, 13, 14, 14, 17, 18], StandardDeviation).

:- end_tests(standard_deviation_test).

:- begin_tests(sum_squares_differences_test).

    test(empty_list, Sum == 0) :-
        sum_squares_differences([], 0, Sum).

    test(single_element_list, Sum == 0) :-
        sum_squares_differences([1], 1, Sum).

    test(multiple_elements_list, Sum == 2) :-
        sum_squares_differences([1,2,3], 2, Sum).

:- end_tests(sum_squares_differences_test).

% [nondet] means that there's multiple cases and they should all be tested.
:- begin_tests(count_occurrences_test).

    test(empty_list, [nondet]) :-
        count_occurrences(a, [], 0).

    test(single_element, [nondet]) :-
        count_occurrences(a, [a], 1).

    test(multiple_elements_present, [nondet]) :-
        count_occurrences(a, [b, a, c, a, d, a], 3).

    test(no_element_present, [nondet]) :-
        count_occurrences(a, [b, c, d], 0).

:- end_tests(count_occurrences_test).

:- begin_tests(classify_test).

    test(same_count, [nondet]) :-
        classify([a, b, c], a).

    test(one_element_list, [nondet]) :-
        classify([a], a).

    test(multiple_most_frequent, [nondet]) :-
        classify([a, b, a, c], a).

:- end_tests(classify_test).

:- begin_tests(get_most_frequent_test).

    test(single_element, [nondet]) :-
        get_most_frequent([3], 3).

    test(first_element, [nondet]) :-
        get_most_frequent([1, 1, 2, 3], 1).

    test(last_element, [nondet]) :-
        get_most_frequent([1, 2, 3, 3], 3).

:- end_tests(get_most_frequent_test).

:- begin_tests(get_classes_test).

    test(empty_input) :-
        get_classes([], []).

    test(single_neighbor) :-
        get_classes([3-'Iris-setosa'], ['Iris-setosa']).

    test(multiple_neighbors) :-
        get_classes([2-'Iris-setosa', 3-'Iris-versicolor', 4-'Iris-setosa'], ['Iris-setosa', 'Iris-versicolor', 'Iris-setosa']).

:- end_tests(get_classes_test).