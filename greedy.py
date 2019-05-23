import numpy as np

class GreedyTSP():
    def __init__(self, n_city):
        Ncity = n_city
        self.cities = Ncity.cities
        self.start_city = Ncity.start_city
        self.init_tour = Ncity.init_tour
        
        self.interval = list()

        self.visited_cities = list()

        self.greedy_tour = list()
        self.distance = lambda x,y: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

    def f(self, point, tour):
        '''
        input: tour (list)

        Function that evaluates the cost of every single remaining node

        output: distance
        '''
        distances = list() # distances is an ordered list containing visited cities
    
        
        for i in range(len(list(tour))):
            if i < len(tour)-1:
                if tour[i] != self.start_city:
                    tour_distance = self.distance(self.cities[point][0], self.cities[tour[i]][0])
                    distances.append((tour[i], tour_distance))

        return distances

    def perform_greedy(self, tour):
        total_distance = 0.0
        city_keys = len(tour)
        next_node = (self.start_city, 0)
        while len(self.visited_cities) != city_keys:
            dist = self.f(next_node[0], tour)
            if len(dist) > 0:
                next_node = min(dist, key=lambda n: (n[1], -n[0]))
                self.visited_cities.append(next_node[0])
                total_distance += next_node[1]
                tour.remove(next_node[0])
            else: # else we are at the last node and return back to starting point
                last_city = self.cities[self.visited_cities[-1]][0]
                begin_city = self.cities[self.start_city][0]
                total_distance += np.sqrt((last_city[0]-begin_city[0])**2 + (last_city[1]-begin_city[1])**2)
                self.visited_cities.append(self.start_city)

        average_distance = total_distance / city_keys
        return total_distance, average_distance
    
    def greedy_this(self):
        tour = self.init_tour.copy()
        total_distance, average_distance = self.perform_greedy(tour)
        return total_distance, average_distance