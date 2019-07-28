import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
import random
# Display all floats rounded off to 1 decimal place
pd.options.display.float_format = '{:,.1f}'.format
# Plot inline in Jupyter notebook
#%matplotlib inline
# Settings throughout the notebook
sns.set()
# Width = 16, Height = 6
DIMS=(16, 6)
import os


class TSPAnneal:

    '''
    Pass the max steps you want to take to the annealer function
    '''

    def __init__(
        self,
#       maxsteps=500,
#       multiplier=1,
#       control_t=1,
#       acceptrate=0.5,
#       lams=0,
#       swaps=round((Ncity.n)**0.5),
#       explore=30,
    accs = [500, 1, 1, 0.5, 0, round((Ncity.n)**0.5), 30]
        ):
        '''
        inputs: total number of steps to try, geometric multiplier for annealing schedule
        
        Initialize parameters
        
        output: none
        '''

        self.cities = Ncity.cities  # self.cities needs to be a
        self.start_city = Ncity.start_city
        self.init_tour = Ncity.generate_initial_tour()

        self.Tmax = accs[0]
        self.threshold = accs[1]  # for geometric scaling
        self.interval = list()
        self.exploration_space = accs[6]

        self.control = accs[2]
        self.trig_lams = accs[4]
        self.acceptrate = accs[3]

        self.swaps = accs[5]

        self.distance = lambda x, y: np.sqrt((x[0] - y[0]) ** 2 + (x[1]
                - y[1]) ** 2)
        self.table_distances = Ncity.table_distances

    def f(self, tour):
        '''
        input: tour (list)
        
        Function that evaluates the cost of a given x1, x2 (euclidean distance)
        
        output: single cost
        '''
        distances = [self.table_distances[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)]

        total_distance = sum(distances)
        average_tour_len = total_distance / len(tour)
        return (total_distance, average_tour_len)

    def acceptance_probability(
        self,
        cost,
        new_cost,
        temperature,
        ):
        '''
        inputs: old cost, new cost, current temperature
        
        calculate probability of acceptance and return it using the metropolis algorithm
        
        output: probability (0 to 1)
        '''

        return np.exp(-(new_cost - cost) / temperature)

    def swap_random(self, tour):
        '''
        randomly swaps 2 tours
        '''

        tour = tour[1:][:-1]
        idx = range(Ncity.n - 1)
        for i in range(self.swaps):
            (i1, i2) = random.sample(idx, 2)
            (tour[i1], tour[i2]) = (tour[i2], tour[i1])

        tour = [self.start_city] + tour + [self.start_city]
        (cost, average_tour_len) = self.f(tour)
        return (tour, cost, average_tour_len)

    def anneal(self):
        '''
        inputs: none
        
        function performs annealing and calls random start to kickstart the annealing process. iteratively
        calculates the new cost.
        
        output: final cost, final state (list of x1 and x2), all costs (list of costs at every timestep)
        
        '''

        # params related to returning the cost and deviation from the optimal objective function
        # deviation = list()

        # params related to Lam's Annealing Schedule

        best_tour = list()
        current_tour = list()
        best_tour_total = list()
        current_tour_total = list()
        tours = list()
        T_list = list()

        acceptrate = self.acceptrate
        LamRate = 0
        tours.append(self.init_tour)

        
        (costs, average_tour_length) = self.f(self.init_tour)
        states = self.init_tour
        best_tour_total.append(costs)
        best_tour.append(average_tour_length)

        for temp_step in range(self.Tmax):

            # for each temperature step
            fraction = temp_step / float(self.Tmax)

            if self.control == 0 & temp_step > 0:
                T = self.threshold * (1 - fraction)
            else:
                T = 1 - fraction

            T_list.append(T)

            # exploration space

            (new_tour, new_cost, new_average_tour_length) = self.swap_random(states)
                
            if new_tour not in tours:
                tours.append(new_tour)

            current_tour_total.append(new_cost)
            current_tour.append(new_average_tour_length)

            if (new_cost < costs) or (self.acceptance_probability(costs,
            new_cost, T) >= random.uniform(0, 1)):
                (states, costs, average_tour_length) = (new_tour, new_cost, new_average_tour_length)

                if self.trig_lams == 1:
                    acceptrate = 1 / 500 * (499 * acceptrate + 1)
            else:

                if self.trig_lams == 1:

                    acceptrate = 1 / 500 * (499 * acceptrate)

                # check conditions

                if fraction < 0.15:
                    LamRate = 0.44 + 0.56 * 560 ** (-temp_step
                    / (self.Tmax * 0.15))
                elif fraction < 0.65:
                    LamRate = 0.44
                else:
                    LamRate = 0.44 * 440 ** ((-temp_step / self.Tmax
                    - 0.65) / 0.35)

                if LamRate < acceptrate:
                    T *= 0.99
                else:
                    T *= 1 / 0.999

            if best_tour_total[-1] > costs:
                best_tour_total.append(costs)
                best_tour.append(average_tour_length)
            else:
                best_tour_total.append(best_tour_total[-1])
                best_tour.append(best_tour[-1])
            
        return (best_tour[1:], current_tour, best_tour_total[1:], current_tour_total, T_list, tours)
