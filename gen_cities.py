from double_linked import doubly_linked_list
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.1f}'.format
import random

class GenCities():
    start_city = 0.0
    init_tour = list()
    cities = dict()
  
    def __init__(self, no_cities=10):
        self.n = no_cities
        self.i1 = np.arange(0, 0+np.sqrt(self.n), 0.01)
        self.i2 = np.arange(0, 0+np.sqrt(self.n), 0.01)

        self.dllist = doubly_linked_list()

        GenCities.start_city = random.randint(0,self.n)
        print(GenCities.start_city)
        self.dllist.push(GenCities.start_city) 
        
    def random_start(self):
        """ 
        input: none

        Randomly choose a point within the boundary

        output: (x1, x2)
        """
        x, y = [random.uniform(self.i1[0], self.i1[-1]), random.uniform(self.i2[0], self.i2[-1])]

        return [x,y]    
  
    def random_neighbour(self):
        """ 
        input: x (a list) containing cities visited
        Choose a random point to move to, and checks whether that point has already been traversed
        output: (newx, newy)
        """

            # normalized 
        unvisited = [i for i in list(GenCities.cities.keys()) if GenCities.cities[i]==0]
        if GenCities.cities[GenCities.start_city][1] < 2:
            unvisited.append(GenCities.start_city)

            # choose a rnadom city to visit (in key form)
        if len(unvisited) > 1:
            choose_city = unvisited[random.randint(0,len(unvisited))]

                # rechoose new city if the chosen city is the start city
        if unvisited == start_city:
            choose_city = unvisited[random.randint(0,len(unvisited))]
        else:
            choose_city = start_city

        GenCities.cities[choose_city][1]+=1
        return GenCities.cities[choose_city]

    def generate_cities(self):
        '''
        reinitializes at a random point
        '''
        city = GenCities.start_city
        state = self.random_start()     
        GenCities.cities[city] = [state, 0]

        for i in range(self.n): # not to include the start city
                    # this ensures we don't generate cities that have already been generated
            while len(list(GenCities.cities.keys())) < self.n:
                if city in list(GenCities.cities.keys()):
                    city = random.randint(1,self.n)
                else:
                    state = self.random_start()
                                    # not calculating the cost
                    GenCities.cities[city] = [state, 0]
                    self.dllist.push(city) 
        return GenCities.cities
    
    def print_city_map(self):
        x1m, x2m = np.meshgrid(self.i1, self.i2)
        fm = np.zeros(x1m.shape)
        for i in range(x1m.shape[0]):
            for j in range(x1m.shape[1]):
                fm[i][j] = 0.2 + x1m[i][j]**2 + x2m[i][j]**2 \
                             - 0.1*math.cos(6.0*3.1415*x1m[i][j]) \
                             - 0.1*math.cos(6.0*3.1415*x2m[i][j])

        plt.figure()
            #CS = plt.contour(x1m, x2m, fm)#,lines)
            #plt.clabel(CS, inline=1, fontsize=10)
        plt.title('TSP Map')
        plt.xlabel('x')
        plt.ylabel('y')

        lists = sorted(GenCities.cities.values()) # sorted by key, return a list of tuples
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        plt.scatter(*zip(*x))
        plt.show()
    
    
    def generate_initial_tour(self):
        tour = list()
        city_copy = GenCities.cities.copy() # create a local copy of city to modify it
        print(city_copy)
            # append the start city to the start and end
        unvisited = [k for k,v in GenCities.cities.items() if GenCities.cities[k][1]==0 and k!=GenCities.start_city]
            # unvisited = [start_city] + unvisited + [start_city]

        #dllist = doubly_linked_list()
        for i in range(len(unvisited)+2):
            unvisit = unvisited.copy()
            if i == (len(unvisited)+1):
                city_copy[GenCities.start_city][1] += 1
                self.dllist.push(GenCities.start_city)

            else:
                next_city = unvisited[random.randint(0, len(unvisited)-1)]

                                # iteratively generate next city
                while (next_city == GenCities.start_city) or (next_city not in unvisit) or (GenCities.cities[next_city][1]==1):
                    next_city = unvisit[random.randint(0, len(unvisit)-1)]
                    if ((next_city != GenCities.start_city) and (next_city not in unvisit)) and (GenCities.cities[next_city][1]==1):
                        unvisit.pop(unvisit.index(next_city))
                    city_copy[next_city][1] += 1
                    self.dllist.push(next_city)

        GenCities.init_tour = self.dllist.get_tour(self.dllist.head)    

        return GenCities.init_tour

# if __name__ == "__main__":
