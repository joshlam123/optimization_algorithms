import math
# %matplotlib qt5
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
# Display all floats rounded off to 1 decimal place
pd.options.display.float_format = '{:,.1f}'.format
sns.set()
# Width = 16, Height = 6
DIMS=(16, 6)
import os

# multiprocessing libraries
from itertools import repeat
from multiprocessing import Pool, Manager
import cProfile
import random
import re    
import functools
import json

# for travelling salesman
import string

# note if you want to change the function, remember to change the boundaries at which the function is evaluated!
class Annealer(object):
    
    '''
    Pass the max steps you want to take to the annealer function
    '''
    
    def __init__(self, custom_function, maxsteps=500, multiplier=1, control_t = 1, acceptrate = 0.5, explore = 30, lams=1, i1=np.arange(-15.0, 10., 0.01), i2=np.arange(-10.0, 10.02, 0.01)):
        '''
        inputs:   
        maxsteps - total number of temperature steps to anneal for (default = 500)
        multiplier - eometric multiplier for annealing schedule (default = 1 OFF)
        control_t - whether you want to turn on or off the geometric cooling schedule (default = 1 OFF)
        acceptrate - generic lam's acceptance rate (default = 0.5)
        explore - number of steps to explore at every iteration (default = 30 steps per iteration)
        lams - whether to turn on or off lam's annealing schedule (default = 1 OFF)
        i1 - range for DV # 1
        i2 - range for DV # 2
        custom_function - no default, but must pass in something

        Initialize parameters
        
        output: none
        '''
        self.Tmax = maxsteps
        self.threshold = multiplier
        self.interval = list()
        self.over_count = 0
        #self.states = {"x":list(), "y":list()}
        self.acceptrate = acceptrate
        self.control = control_t
        self.exploration_space = explore
        self.costs = dict()
        self.deviations = dict()
        self.trig_lams = lams
        self.real_answer = 0
        self.lams = dict()
        self.accepts = dict()
        self.T = list()
        self.i1 = i1
        self.i2 = i2
        self.function = custom_function

        
    def random_start(self):
        """ 
        input: none
        
        Randomly choose a random starting point within the boundary
        
        output: a pair of starting point coordinates (x1, x2)
        """
        self.interval.append([random.uniform(self.i1[0], self.i1[-1]), random.uniform(self.i2[0], self.i2[-1])])
        #self.states["x"].append(self.interval[0])
        #self.states["y"].append(self.interval[1])
        return self.interval
    
    def f(self, x):
        '''
        input: x (a 2D array)
        
        Function that evaluates the cost of a given x1, x2
        
        output: single cost
        '''

        x1 = x[0]
        x2 = x[1]
        
        obj = eval(self.function)
        '''levy function n.13 - multimodal function'''
        #obj = (np.sin(3*np.pi*x[0])**2) + (((x[0]-1)**2) * (1+(np.sin(3*np.pi*x[1]))**2)) + (((x[1]-1)**2) * (1+(np.sin(2*np.pi*x[1]))**2))
        
        '''bukin function n.6 - many local minima'''
        #obj = 100 * np.sqrt(abs(x[1] - 0.01*(-x[0])**2)) + 0.01 * abs(x[0] + 10)
        
        '''griewank function - many local minima'''
        #obj = ((x[0]**2+x[1]**2) / 4000) - (np.cos(x[0])*np.cos(x[1])) + 1
        
        '''eggholder function - many local minima'''
        #obj = (-(x[1]+47)*np.sin(np.sqrt(abs(x[1] + (x[0]/2) + 47)))) - (x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]+47)))))
        
        return obj

    def random_neighbour(self, x):
        """ 
        input: x (a 2D array)
        
        Move a little bit x1 and x2, from the left or the right and then check whether it's within
        the boundary. (normalized by the min and max) 
        if it's within the boundary, return the new coordinates, otherwise find new ones.
        
        output: (newx, newy)
        """
        
        # normalized 
        deltax = random.uniform(self.i1[0], self.i1[-1]) 
        deltay = random.uniform(self.i2[0], self.i2[-1])
        
        newx = x[0] + deltax
        newy = x[1] + deltay
                             
        return [newx, newy]
    
    def acceptance_probability(self, cost, new_cost, temperature):
        '''
        inputs: old cost, new cost, current temperature
        
        calculate probability of acceptance and return it using the metropolis algorithm
        
        output: probability (0 to 1)
        '''
        return min(1, np.exp((-(new_cost - cost)) / temperature))
    
    def restart(self):
        '''
        reinitializes at a random point
        '''
        state = self.random_start()[0]
        cost = self.f(state)
        return state, cost
    
    def anneal(self):
        '''
        inputs: none
        
        function performs annealing and calls random start to kickstart the annealing process. iteratively
        calculates the new cost.
        
        output: final cost, final state (list of x1 and x2), all costs (list of costs at every timestep)
        '''
        current_cost = list()
        deviation = list()
        acceptrate = self.acceptrate
        states, costs = self.restart()
        LamRate = 0
        
        for temp_step in range(self.Tmax): 
            current_cost = list() 

            fraction = temp_step / float(self.Tmax)
            
            
            if self.control == 0 & temp_step > 0:
                T = self.threshold * (1-fraction)
            else:
                T = 1 - fraction
            
            if self.trig_lams == 0:
                if temp_step < 1:
                    T = 0.5

            for step in range(self.exploration_space): 

                new_state = self.random_neighbour(states)
                new_cost = self.f(new_state)
                #current_cost.append(new_cost)
                
                if new_cost < costs:
                    states, costs = new_state, new_cost
                    current_cost.append(costs)
                    if self.trig_lams == 1:
                        acceprate = (1/500) * (499 * acceptrate + 1)
                else:
                    if self.trig_lams == 0:
                        if self.acceptance_probability(costs, new_cost, T) >= random.uniform(0,1):
                            states, costs = new_state, new_cost
                            
                    else:
                        '''trigger lam's function, use this annealing shcedule instead'''
                        if self.acceptance_probability(costs, new_cost, T) >= random.uniform(0,1):
                            states, costs = new_state, new_cost
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
            
            if self.trig_lams == 1:
                if temp_step not in list(self.lams.keys()):
                    self.lams[temp_step] = list()
                if temp_step not in list(self.accepts.keys()):
                    self.accepts[temp_step] = list()
                self.lams[temp_step].append(LamRate)
                self.accepts[temp_step].append(acceptrate)
                self.T.append(T)
            
            current_cost.append(costs)
            deviation.append(abs(costs-self.real_answer))

            if T not in list(self.costs.keys()):
                self.costs[T] = list()
            self.costs[T].append(current_cost)
            if T not in list(self.deviations.keys()):
                self.deviations[T] = list()
            self.deviations[T].append(deviation)
        return self.costs, self.deviations, self.accepts, self.lams, self.T


def get_range():
    Bukin = '100 * np.sqrt(abs(x[1] - 0.01*(-x[0])**2)) + 0.01 * abs(x[0] + 10)'
    function_choice = input('Please input your desired function, e.g. Bukin Function n.6, 100 * np.sqrt(abs(x[1] - 0.01*(-x[0])**2)) + 0.01 * abs(x[0] + 10) \n')
    i1 = input('Please input desired x1 range in the form x1,y1: \n')
    i2 = input('Please input desired x1 range in the form x1,y1: \n')
    if function_choice == "":
        function_choice = Bukin
        i1 = [15.0, -10.0]
        i2 = [15.0, -10.0]
    else:
        special_chars = r'[`\=~!@#$%^&*()_+\[\]{};\'\\:"|<,/<>?]'
      
        i1, i2 = re.split(special_chars, i1), re.split(special_chars, i2)
        i1, i2 = [np.float(i) for i in i1], [np.float(i) for i in i1]
    i1 = np.arange(min(i1), max(i1), 0.01)
    i2 = np.arange(min(i2), max(i2), 0.01)

    return function_choice, i1, i2


if __name__ == '__main__':
	
	# add annealing code here
    ''' THIS IS SAMPLE CODE '''
    function_choice, r1, r2 = get_range()
    tries_temp = {"run":list(), "temp":list(), "cost":list(), "deviations":list()}
    a = Annealer(custom_function=function_choice, maxsteps=5000, multiplier=1, control_t=1, i1=r1, i2=r2)
    costs, deviations, accepts, lams, T = a.anneal()
    cost_keys = list(costs.keys())
    for k in cost_keys:
        tries_temp['run'].append(1)
        tries_temp['temp'].append(k)
        if len(costs[k][-1]) == 0:
            tries_temp['cost'].append(0)
        else:
            tries_temp['cost'].append(costs[k][-1][-1])
            tries_temp['deviations'].append(deviations[k][-1][-1])
	#cost_keys = list(costs.keys())
    print(costs)
    temp_df = pd.DataFrame.from_dict(tries_temp)

    plt.xlabel("Temperature")
    plt.ylabel("Cost", fontsize=12)
    plt.xlim(0.5,0)
    plt.ylim(0,30)
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1.01, 11))))
    plt.plot(temp_df['temp'].tolist(), temp_df['cost'].tolist())
    plt.plot(temp_df['temp'].tolist(), temp_df['deviations'].tolist())
    plt.show()
    ''' END OF SAMPLE CODE ''' 