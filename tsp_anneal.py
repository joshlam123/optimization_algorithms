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

# note if you want to change the function, remember to change the boundaries at which the function is evaluated!
class TSP_Anneal():
    
    '''
    Pass the max steps you want to take to the annealer function
    '''
    def __init__(self, n_city, maxsteps=500, multiplier=1, control_t = 1, acceptrate = 0.5, lams=0, swaps=1, explore = 30):
        '''
        inputs: total number of steps to try, geometric multiplier for annealing schedule
        
        Initialize parameters
        
        output: none
        '''
        Ncity = n_city
        self.cities = Ncity.cities # self.cities needs to be a 
        self.start_city = Ncity.start_city
        self.init_tour = Ncity.init_tour
        
        self.Tmax = maxsteps
        self.threshold = multiplier # for geometric scaling
        self.interval = list()
        self.exploration_space = explore
        
        self.control = control_t
        self.trig_lams = lams
        self.acceptrate = acceptrate
        #self.lams = dict()
        #self.accepts = dict()
        #self.deviations = dict() # only when you have brute force
        
        self.swaps = swaps
        
        self.tours = dict()
        self.avg_len = dict()
        self.tour_len = dict()
        self.distance = distance = lambda x,y: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

    
    def f(self, tour):
        '''
        input: tour (list)
        
        Function that evaluates the cost of a given x1, x2 (euclidean distance)
        
        output: single cost
        '''
        total_distance = 0.0 
        
        distances = [self.distance(self.cities[tour[i]][0], self.cities[tour[i+1]][0]) for i in range(len(tour) - 1)]
#         for i in range(len(tour)):
#             if i < len(tour)-1:
                
#                 total_distance += distance(self.cities[tour[i]][0], self.cities[tour[i+1]][0])
        total_distance = sum(distances)
        average_tour_len = total_distance / len(tour)
        return total_distance, average_tour_len
    
    def acceptance_probability(self, cost, new_cost, temperature):
        '''
        inputs: old cost, new cost, current temperature
        
        calculate probability of acceptance and return it using the metropolis algorithm
        
        output: probability (0 to 1)
        '''
        return min(1, np.exp((-(new_cost - cost)) / temperature))
    
    def swap_random(self, tour):
        tour = tour[1:][:-1] 
        idx = range(len(tour))
        for i in range(self.swaps):
            i1, i2 = random.sample(idx, 2)
            tour[i1], tour[i2] = tour[i2], tour[i1]
            
        tour = [self.start_city] + tour + [self.start_city]
        cost, average_tour_len = self.f(tour)
        return tour, cost, average_tour_len
    
    def anneal(self):
        '''
        inputs: none
        
        function performs annealing and calls random start to kickstart the annealing process. iteratively
        calculates the new cost.
        
        output: final cost, final state (list of x1 and x2), all costs (list of costs at every timestep)
        
        TODO: Implement a Greedy, Brute Force Algorithm, Calculate the Optimal Tour Length Ratio 
        '''
        
        
        # params related to returning the cost and deviation from the optimal objective function 
        # deviation = list()

        # params related to Lam's Annealing Schedule
        acceptrate = self.acceptrate
        LamRate = 0
        
        costs, average_tour_length = self.f(self.init_tour)
        states = self.init_tour
        self.avg_len[1] = average_tour_length
        self.tours[1] = states
        self.tour_len[1] = costs
        
        for temp_step in range(self.Tmax): 

            current_cost = list() 
            
        
            # for each temperature step
            fraction = temp_step / float(self.Tmax)
            
            if self.control == 0 & temp_step > 0:
                T = self.threshold * (1-fraction)
            else:
                T = 1 - fraction
            
            if self.trig_lams == 0:
                if temp_step < 1:
                    T = 0.5
            
            for step in range(self.exploration_space): # make sure we generate all the points first before calculating cost

                new_tour, new_cost, new_average_tour_length = self.swap_random(states)
                
                if new_tour in [self.tours[k] for k,v in self.tours.items()]:
                    pass
                #current_cost.append(new_cost)
                
                else:
                    if new_cost < costs:
                        states, costs, average_tour_length = new_tour, new_cost, new_average_tour_length

                        if self.trig_lams == 1:
                            acceprate = (1/500) * (499 * acceptrate + 1)
                    else:
                        if self.trig_lams == 0:
                            if self.acceptance_probability(costs, new_cost, T) >= random.uniform(0,1):
                                states, costs, average_tour_length = new_tour, new_cost, new_average_tour_length

                        else:
                            '''trigger lam's function, use this annealing shcedule instead'''
                            if self.acceptance_probability(costs, new_cost, T) >= random.uniform(0,1):
                                states, costs, average_tour_length = new_tour, new_cost, new_average_tour_length
                                #current_cost.append(costs)
                                acceptrate = (1/500) * (499 * acceptrate + 1)
                            else:
                                acceptrate = (1/500) * (499 * acceptrate)

                            # check conditions
                            if fraction < 0.15:
                                LamRate = 0.44 + 0.56 * 560**((-temp_step)/(self.Tmax*0.15))
                            elif fraction < 0.65:
                                LamRate = 0.44
                            else:
                                LamRate = 0.44 * 440**((((-temp_step)/self.Tmax)-0.65)/0.35)

                                    # compare 2 rates
                            if LamRate < acceptrate:
                                T *= 0.99
                            else:
                                T *= 1/0.999
                            
#             if self.trig_lams == 0:
#                 if temp_step not in list(self.lams.keys()):
#                     self.lams[temp_step] = list()
#                 if temp_step not in list(self.accepts.keys()):
#                     self.accepts[temp_step] = list()
#                 self.lams[temp_step].append(LamRate)
#                 self.accepts[temp_step].append(acceptrate)
#                 self.T.append(T)
            
            #deviation.append(abs(costs-self.real_answer)) # only when you have brute force
            
        
            self.avg_len[T] = average_tour_length
            self.tours[T] = states
            self.tour_len[T] = costs
#             if T not in list(self.deviations.keys()):
#                 self.deviations[T] = list()
#             self.deviations[T].append(deviation)
        return self.tour_len, self.avg_len #self.deviations, self.accepts, self.lams, self.T

