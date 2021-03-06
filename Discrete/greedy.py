import numpy as np

#!/usr/bin/python
# -*- coding: utf-8 -*-


class GreedyTSP:

    def __init__(self, cities, start_city, table_distances):
        self.cities = cities
        self.start_city = start_city
        self.interval = list()

        self.visited_cities = list()

        self.greedy_tour = list()
        self.distance = lambda x, y: np.sqrt((x[0] - y[0]) ** 2 + (x[1]
                - y[1]) ** 2)
        self.table_distances = table_distances


    def f(self, point):
        '''
        input: a point

        Function that finds the minimum distance based on the table with distances to each point

        returns: the distance to the next node
        '''

        next_node = min(self.table_distances[point],
                        key=self.table_distances[point].get)
        distance = self.table_distances[point][next_node]
        return (next_node, distance)

    def perform_greedy(self):
        '''
        function that calls f to perform the greedy function and calculate the distance
        '''

        total_distance = 0.0
        city_keys = len(self.cities.keys())
        next_node = (self.start_city, 0)
        self.visited_cities.append(self.start_city)
        while len(self.visited_cities) <= city_keys:
            # check whether we are at the last node

            if len(self.visited_cities) <= city_keys-1:
                next_node = self.f(next_node[0])

                while next_node[0] == self.start_city:
                    next_node = self.f(next_node[0])
                    
                self.visited_cities.append(next_node[0])

                for (k, v) in self.table_distances.items():
                    if k != next_node[0]:
                        del self.table_distances[k][next_node[0]]
                total_distance += next_node[1]
                
            else:

            # else we are at the last node and return back to starting point
                total_distance += \
                    self.table_distances[next_node[0]][self.start_city]
                
                self.visited_cities.append(self.start_city)

            avg = total_distance / len(self.visited_cities)

        average_distance = total_distance / city_keys
        return (total_distance, average_distance)

    def greedy_this(self):
        '''
        call this function to run the entire annealing algortihm
        '''
        (total_distance, average_distance) = self.perform_greedy()
        return (total_distance, average_distance)
    