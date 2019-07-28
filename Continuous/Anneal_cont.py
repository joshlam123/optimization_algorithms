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
import random


#!/usr/bin/python
# -*- coding: utf-8 -*-
# note if you want to change the function, remember to change the boundaries at which the function is evaluated!


class Annealer(object):
    '''
    Pass the max steps you want to take to the annealer function
    '''

    def __init__(
        self,
        maxsteps=500,
        multiplier=1,
        control_t=1,
        acceptrate=0.5,
        explore=30,
        lams=1,
        i1=np.arange(-15.0, 10., 0.01),
        i2=np.arange(-10., 10.02, 0.01),
        ):
        '''
        inputs:   
        maxsteps - total number of temperature steps to anneal for (default = 500)
        multiplier - eometric multiplier for annealing schedule (default = 1 OFF)
        control_t - whether you want to turn on or off the geometric cooling schedule (default = 1 OFF)
        acceptrate - generic lam's acceptance rate (default = 0.5)
        explore - number of steps to explore at every iteration (default = 30 steps per iteration)
        lams - whether to turn on or off lam's annealing schedule (default = 1 OFF)
        
        Initialize parameters
        
        output: none
        '''

        self.Tmax = maxsteps
        self.threshold = multiplier
        self.interval = list()
        self.over_count = 0

        # self.states = {"x":list(), "y":list()}

        self.acceptrate = acceptrate
        self.control = control_t
        self.exploration_space = explore
        self.trig_lams = lams
        self.real_answer = -1.8013
        self.lams = dict()
        self.accepts = dict()
        self.i1 = i1
        self.i2 = i2

    def get_range(self):
        '''
        function to get range from the user
        '''
        i1 = input('Please input desired x1 range in the form x1,y1: \n'
                   )
        i2 = input('Please input desired x1 range in the form x1,y1: \n'
                   )

        special_chars = r'[`\=~!@#$%^&*()_+\[\]{};\'\\:"|<,/<>?]'

        (i1, i2) = (re.split(special_chars, i1),
                    re.split(special_chars, i2))
        (i1, i2) = ([np.float(i) for i in i1], [np.float(i) for i in
                    i1])
        i1 = np.arange(min(i1), max(i1), 0.01)
        i2 = np.arange(min(i2), max(i2), 0.01)

        return (i1, i2)

    def random_start(self):
        """ 
        input: none
        
        Randomly choose a random starting point within the boundary
        
        output: a pair of starting point coordinates (x1, x2)
        """

        self.interval.append([random.uniform(self.i1[0], self.i1[-1]),
                             random.uniform(self.i2[0], self.i2[-1])])

        return self.interval


    def f(self, x):
        '''
        input: x (a 2D array)
        
        Function that evaluates the cost of a given x1, x2
        
        output: single cost
        '''

        x1 = x[0]
        x2 = x[1]
    
    # function 1, levy function 
        obj = np.sin(3 * np.pi * x[0]) ** 2 + (x[0] - 1) ** 2 * (1
                + np.sin(3 * np.pi * x[1]) ** 2) + (x[1] - 1) ** 2 * (1
                + np.sin(2 * np.pi * x[1]) ** 2)

        # self.i1 = np.arange(-10.0, 10., 0.01)
        # self.i2 = np.arange(-10.0, 10., 0.01)

        # obj = 100 * np.sqrt(abs(x[1] - 0.01*(-x[0])**2)) + 0.01 * abs(x[0] + 10)
        # self.i1 = np.arange(-15.0, 10., 0.01)
        # self.i2 = np.arange(-15.0, 10., 0.01)

        #obj = - ((np.sin(x[1])* (np.sin((x[1]**2) / (np.pi))**20 )) + (np.sin(x[1])*(np.sin(2*(x[1]**2) / (np.pi))**20 )))
        # self.i1 = np.arange(0, np.pi, 0.01)
        # self.i2 = np.arange(0, np.pi, 0.01)

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


    def restart(self):
        '''
        reinitializes at a random point
        '''

        state = self.random_start()[0]
        cost = self.f(state)
        return (state, cost)


    def anneal(self):
        '''
        inputs: none
        
        function performs annealing and calls random start to kickstart the annealing process. iteratively
        calculates the new cost.
        
        output: final cost, final state (list of x1 and x2), all costs (list of costs at every timestep)
        '''

        best_cost = list()
        current_cost = list()
        deviation = list()
        T_list = list()
    
        acceptrate = self.acceptrate
        (states, costs) = self.restart()
        LamRate = 0

        best_cost.append(costs)

        for temp_step in range(self.Tmax):
            

            fraction = temp_step / float(self.Tmax)

        # T = max((1-self.trig_lams) * max(fraction*(1-self.control), (1 - fraction) * self.control) * self.threshold, (1-fraction)*self.trig_lams)

        # if you want to trigger lam's, self.control == 1

            if self.control == 0 & temp_step > 0:
                T = self.threshold * (1 - fraction)
            else:
                T = 1 - fraction

            T_list.append(T)

            for step in range(self.exploration_space):

                new_cost = costs
                new_state = states

                gen_new_state = self.random_neighbour(new_state)
                gen_new_cost = self.f(gen_new_state)

                if gen_new_cost < new_cost:
                    new_state = self.random_neighbour(states)
                    new_cost = self.f(new_state)

            current_cost.append(new_cost)

            if new_cost < costs or self.acceptance_probability(costs,
                new_cost, T) >= random.uniform(0, 1):
                states, costs = new_state, new_cost

            if self.trig_lams == 1:
                acceprate = 1 / 500 * (499 * acceptrate + 1)
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
                        LamRate = 0.44 * 440 ** ((-temp_step
                          / self.Tmax - 0.65) / 0.35)


                    if LamRate < acceptrate:
                      T *= 0.99
                    else:
                      T *= 1 / 0.999

            
            deviation.append(abs(costs - self.real_answer))

            if best_cost[-1] > costs:
                best_cost.append(costs)
            else:
                best_cost.append(best_cost[-1])

            if self.trig_lams == 1:
                if temp_step not in list(self.lams.keys()):
                    self.lams[temp_step] = list()
                if temp_step not in list(self.accepts.keys()):
                    self.accepts[temp_step] = list()
                self.lams[temp_step].append(LamRate)
                self.accepts[temp_step].append(acceptrate)


        return (
            current_cost,
            best_cost[1:],
            deviation,
            self.accepts,
            self.lams,
            T_list,
            )


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
    tries1 = {"run":list(), "temp":list(), "current_cost":list(), "best_cost":list(), "deviations":list()}

    for i in tqdm(range(0, 100, 1)):
        a = Annealer(custom_function=function_choice, maxsteps=5000, multiplier=1, control_t=1, i1=r1, i2=r2)
        current_cost, best_cost, deviations, accepts, lams, T = a.anneal()
        cost_keys = len(list(current_cost))
        for k in range(cost_keys):
            tries1['run'].append(i)
            tries1['temp'].append(T[k])
            tries1['current_cost'].append(current_cost[k]) 
            tries1['best_cost'].append(best_cost[k])
            tries1['deviations'].append(deviations[k])

    ''' converts the dictionary into a pandas dataframe for easy data manipulation'''
    df_case1 = pd.DataFrame.from_dict(tries1)
    #df_case1 = df_case1.reindex(index=df_case1.index[::-1])
    df_case1.head(20)
    df_case1_group_mean = df_case1.groupby(['temp']).mean().reset_index()
    df_case1_group_mean.to_csv("case1_func3.csv")


    # TO PLOT TEMPERATURE V. COST
    fig, ax1 = plt.subplots(1, 1)

    plt.xlabel("Temperature")
    plt.ylabel("Cost", fontsize=12)

            #Add the legend
    plt.title("Temperature v. Cost (1 - Ti / Tmax)")
    plt.xlim(1.0, 0)
    #plt.ylim(0,100)
    plt.plot(df_case1_group_mean['temp'].tolist(), df_case1_group_mean['current_cost'].tolist(), label='current_cost')
    plt.plot(df_case1_group_mean['temp'].tolist(), df_case1_group_mean['best_cost'].tolist(), label='best_cost')
    plt.legend(fontsize=12)
    plt.savefig('case_1_costs.png')

    # TO PLOT DEVIATIONS
    fig, ax1 = plt.subplots(1, 1)

    plt.xlabel("Temperature")
    plt.ylabel("Deviations", fontsize=12)

        #Add the legend
    plt.title("Temperature v. Deviation (1 - Ti / Tmax)")
    plt.xlim(1.0, 0)
    plt.plot(df_case1_group_mean['temp'].tolist(), df_case1_group_mean['deviations'].tolist(), label='mean')
    plt.savefig('case_1_deviations.png')

    plt.show()
    ''' END OF SAMPLE CODE ''' 