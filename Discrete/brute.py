#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

#!/usr/bin/python
# -*- coding: utf-8 -*-


class BruteTSP:

    def __init__(self):
        self.cities = Ncity.cities
        self.start_city = Ncity.start_city
        self.init_tour = Ncity.init_tour

        self.interval = list()

        self.visited_cities = list()

        self.greedy_tour = list()
        self.distance = lambda x, y: np.sqrt((x[0] - y[0]) ** 2 + (x[1]
                - y[1]) ** 2)
        self.table_distances = Ncity.table_distances
        self.shortest_distance = list()  # tour: distance
        self.all_distance = list()

    def f(self, tour):
        '''
        input: tour (list)

        Function that evaluates the cost of every single remaining node

        output: distance
        '''

        distances = [self.table_distances[tour[i]][tour[i + 1]]
                     for i in range(len(tour) - 1)]

#         for i in range(len(tour) - 1):
#             total_distance += self.table_distances[tour[i]][tour[i+1]]

        total_distance = sum(distances)
        average_tour_len = total_distance / len(tour)
        return (total_distance, average_tour_len)

    def heap_perm(self, A):
        ''' instantiate the heap algorithm '''

        n = len(A)
        Alist = [el for el in A]
        for hp in self._heap_perm_(n, Alist):
            yield hp

    def _heap_perm_(self, n, A):
        ''' implement the heap algorithm for generating permutations '''

        if n == 1:
            yield A
        else:
            for i in range(n - 1):
                for hp in self._heap_perm_(n - 1, A):
                    yield hp
                j = (0 if n % 2 == 1 else i)
                (A[j], A[n - 1]) = (A[n - 1], A[j])
            for hp in self._heap_perm_(n - 1, A):
                yield hp

    def brute_this(self):
        '''
        generates a tour and adds the shortest distance. instead of generating many permutations, 
        how do we know that we have explored all permutations?? 
        '''


        minimum_distance = (self.init_tour, self.f(self.init_tour))  # initial tour, total, average length

        # perms = list(permutations(self.init_tour[1:][:-1]))

        for item in self.heap_perm(self.init_tour[:1][:-1]):
            self.shortest_distance.append(minimum_distance[1])

            new_tour = [self.start_city] + item + [self.start_city]
            cost = (item, self.f(new_tour))

            self.all_distance.append(cost)

            if minimum_distance[1] > cost:  # if new tour cost is lesser than the cost of the old tour
                minium_distance = (new_tour, cost)  # gen permutation

#         for i in perms:
#             tours.append([self.start_city] + list(i) + [self.start_city])
        # tours = [[self.start_city] + i + [self.start_city] for i in perms]
#        distances = [self.f(i) for i in tours]
#        total_distance, average_distance = zip(*distances)

#        lowest = min(enumerate(average_distance), key=itemgetter(1))[0]

        return minimum_distance
