from double_linked import doubly_linked_list
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.1f}'.format
import random

#!/usr/bin/python
# -*- coding: utf-8 -*-


class GenCities:

    def __init__(self, no_cities=10):
        self.start_city = 0.0
        self.cities = dict()
        self.n = no_cities

        self.i1 = np.arange(0, 0 + (self.n**0.5), 0.01)
        self.i2 = np.arange(0, 0 + (self.n**0.5), 0.01)
        #self.i1 = np.arange(0, 0 + self.n, 0.01)
        #self.i2 = np.arange(0, 0 + self.n, 0.01)
        self.distance = lambda x, y: np.sqrt((x[0] - y[0]) ** 2 + (x[1]
                - y[1]) ** 2)
        self.dllist = doubly_linked_list()

        self.start_city = random.randint(0, self.n)
        #print(self.start_city)
        self.dllist.push(self.start_city)
        self.table_distances = dict()

        self.generate_cities()
        # self.generate_initial_tour()
        self.precomp_distances()


    def random_start(self):
        """ 
        input: none

        Randomly choose a point within the boundary

        output: (x1, x2)
        """

        (x, y) = [random.uniform(self.i1[0], self.i1[-1]),
                  random.uniform(self.i2[0], self.i2[-1])]

        return [x, y]

    def random_neighbour(self):
        """ 
        input: x (a list) containing cities visited
        Choose a random point to move to, and checks whether that point has already been traversed
        output: (newx, newy)
        """

            # normalized

        unvisited = [i for i in list(self.cities.keys())
                     if self.cities[i] == 0]
        if self.cities[self.start_city][1] < 2:
            unvisited.append(self.start_city)

            # choose a rnadom city to visit (in key form)

        if len(unvisited) > 1:
            choose_city = unvisited[random.randint(0, len(unvisited))]

                # rechoose new city if the chosen city is the start city

        if unvisited == start_city:
            choose_city = unvisited[random.randint(0, len(unvisited))]
        else:
            choose_city = start_city

        self.cities[choose_city][1] += 1
        return self.cities[choose_city]


    def generate_cities(self):
        '''
        reinitializes at a random point
        '''

        city = self.start_city
        state = self.random_start()
        self.cities[city] = [state, 0]

        for i in range(self.n):  # not to include the start city

                    # this ensures we don't generate cities that have already been generated

            while len(list(self.cities.keys())) < self.n:
                if city in list(self.cities.keys()):
                    city = random.randint(1, self.n)
                else:
                    state = self.random_start()

                                    # not calculating the cost

                    self.cities[city] = [state, 0]

                    self.dllist.push(city)
        return self.cities

    def print_city_map(self):
        (x1m, x2m) = np.meshgrid(self.i1, self.i2)
        fm = np.zeros(x1m.shape)
        for i in range(x1m.shape[0]):
            for j in range(x1m.shape[1]):
                fm[i][j] = 0.2 + x1m[i][j] ** 2 + x2m[i][j] ** 2 - 0.1 \
                    * math.cos(6.0 * 3.1415 * x1m[i][j]) - 0.1 \
                    * math.cos(6.0 * 3.1415 * x2m[i][j])

        plt.figure()

            # CS = plt.contour(x1m, x2m, fm)#,lines)
            # plt.clabel(CS, inline=1, fontsize=10)

        plt.title('TSP Map')
        plt.xlabel('x')
        plt.ylabel('y')

        lists = sorted(self.cities.values())  # sorted by key, return a list of tuples
        (x, y) = zip(*lists)  # unpack a list of pairs into two tuples
        plt.scatter(*zip(*x))
        plt.show()

    def generate_initial_tour(self):

        city_copy = self.cities.copy()  # create a local copy of city to modify it

            # append the start city to the start and end

        unvisited = [k for (k, v) in self.cities.items()
                     if k != self.start_city]
        random.shuffle(unvisited)
        tour = [self.start_city] + unvisited + [self.start_city] 
        return tour
      
    def precomp_distances(self):
        '''
        input: none
        function that comes up with a table of distances of each city to the next 
        output: none
        '''

        # distances is an ordered list containing visited cities
        "Cities Precomputed and stored in Matrix"
        for (k, v) in self.cities.items():
            self.table_distances[k] = dict()
            tour_distances = ((i, self.distance(self.cities[i][0],
                               self.cities[k][0])) for (i, j) in
                self.cities.items() if i != k)
            self.table_distances[k] = dict((x, y) for (x, y) in
                    tour_distances)
        return self.table_distances
# def spawn_cities(no_cities, **args):
#     create_city = GenCities(no_cities)
#     create_city.generate_cities()
#     tour = create_city.generate_initial_tour()
#     cities = create_city.cities
#     start_city = create_city.start_city
#     create_city.precomp_distances()
    
#     return create_city

# city = 10
# Ncity = spawn_cities(no_cities=city)

