import numpy as np
import pandas as pd
import math
import random
import math
import scipy.stats
import sys
import time
# this inserts the continuous file into the path
# sys.path.insert(0, '../Simulated Annealing/Continuous')
#from Anneal_cont import Annealer
#!/usr/bin/python
# -*- coding: utf-8 -*-
# note if you want to change the function, remember to change the boundaries at which the function is evaluated!

i1=np.arange(-10., 10.02, 0.01),
i2=np.arange(-10., 10.02, 0.01),

class PAAnneal:

    '''
    Pass the max steps you want to take to the annealer function
    '''

    def __init__(
          self,
        maxsteps=500,
        explore=30,
        walkers=10,
        error_thres=10e-2, 
        correct=0.0,
        multiplier=1, # by default the multipler is 1 
        acceptrate=0.5,
        lams=0, # by default lams is turned off
        choice='multinomial'
      #accs = [500, 1, 1, 0.5, 0, round((Ncity.n)**0.5), 30]
          ):
        '''
        inputs: total number of steps to try, geometric multiplier for annealing schedule
        Initialize parameters
        output: none
        '''
        self.lams, self.acceptrate, self.multiplier = lams, acceptrate, multiplier 
        self.Tmax, self.exploration_space = maxsteps, explore
        self.interval = list()

        self.correct_answer, self.error_threshold, self.cumulative_correct = correct, error_thres, 0.0
        self.i1, self.i2 = i1, i2
        
        self.walkers_t1, self.walkers_t2, self.initial = walkers, walkers, walkers
        
        self.stat_weight_ratio = dict()
        self.partition_function = 0
        
        self.energy_landscape = dict()
        
        # e_diff is a lambda function used to calculate the ratio of statistical weight
        self.e_diff = lambda x, y: np.exp(-(x[1] - x[0]) * y) 
    
    def resample_population(self, walker_pos, mean_val, stat_weight_ratio, Q, tau, choice='poisson '):
        '''
        input: a walker point
        randomly resample the population N times for each replica, where N is a poisson random variable
        output: either a list of samples or None.
        '''

        rv = dict()
        
        if choice == "poisson":
            # current number of replicas over the previous number of replicas
            tau = {k:(self.initial / mean_val * v) for k,v in tau.items()}
            # generate a list of poisson values based on the array
            rv = {k:np.random.poisson(v) for k,v in tau.items()}
        
        else:

            taus = np.array(list(tau.values()))
            normalized_taus = taus / np.sum(taus)
            nums = np.random.multinomial(self.initial, normalized_taus)
            rv = {k:nums[k] for k in range(len(walker_pos))} # this is not self.initial, this is something else. 

        return rv

    def partition_calc(self, walker_pos, t0, t1, mean_val):
        '''
        input: None
        calculate the statistical weight of a single walker, and also 
        output: parition function and statisticla weight ratios for each walker
        '''
        stat_weight_ratio = dict()
        walk_energies = list()
        
        # 1 iteration
        for k,v in walker_pos.items():
            energy = walker_pos[k][1]
            #self.walker_pos[k][1] = energy # append the cost function the walker's position
            swr = self.e_diff([t0, t1], energy) 
            # potential problem here in when we need to reinstantiate
            if k not in stat_weight_ratio.keys():
                stat_weight_ratio[k] = 0.0
            stat_weight_ratio[k] = swr
            
            walk_energies.append(swr)
            
        partition_function = np.sum([np.exp(-(t1) * i[1]) for i in list(walker_pos.values())])
        Q = np.sum(walk_energies) / mean_val
        tau = {k:stat_weight_ratio[k]/Q for k,v in walker_pos.items()}

        return stat_weight_ratio, partition_function, Q, tau


    def max_key(self, walker_pos):
        '''
        inputs: none
        finds the minimum value in the dictionary of walkers
        outputs: key of the lowest (best) cost value in the entire dictionary of walkers
        '''
        return min(walker_pos.keys(), key=(lambda k: walker_pos[k][1]))      

    def random_neighbour(self, x):
        """ 
        input: x (a 2D array)
        
        Move a little bit x1 and x2, from the left or the right and then check whether it's within
        the boundary. (normalized by the min and max) 
        if it's within the boundary, return the new coordinates, otherwise find new ones.
        
        output: (newx, newy)
        """

        # normalized

        deltax = random.uniform(self.i1[0][0], self.i1[0][-1])
        deltay = random.uniform(self.i2[0][0], self.i2[0][-1])

        newx = x[0] + deltax
        newy = x[1] + deltay

        return [newx, newy]
    
    def f(self, x):
        '''
        input: tour (list)

        Function that evaluates the cost of a given x1, x2 (euclidean distance)

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

        #obj = 100 * np.sqrt(abs(x[1] - 0.01*(-x[0])**2)) + 0.01 * abs(x[0] + 10)
        # self.i1 = np.arange(-15.0, 10., 0.01)
        # self.i2 = np.arange(-15.0, 10., 0.01)

        #obj = - ((np.sin(x[1])* (np.sin((x[1]**2) / (np.pi))**20 )) + (np.sin(x[1])*(np.sin(2*(x[1]**2) / (np.pi))**20 )))
        # self.i1 = np.arange(0, np.pi, 0.01)
        # self.i2 = np.arange(0, np.pi, 0.01)

        return obj

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
    
    def check_correct(self, energy):
        self.cumulative_correct += np.sum([1 if (i-self.correct_answer)<=self.error_threshold or i<self.correct_answer else 0 for i in energy])
    
    def max_key(self, walker_pos):
        '''
        inputs: none
        finds the minimum value in the dictionary of walkers
        outputs: key of the lowest (best) cost value in the entire dictionary of walkers
        '''
        return min(walker_pos.keys(), key=(lambda k: walker_pos[k][1]))

    def anneal(self):
        '''
        inputs: none

        function performs annealing and calls random start to kickstart the annealing process. iteratively
        calculates the new cost.

        output: final cost, final state (list of x1 and x2), all costs (list of costs at every timestep)

        '''
        T_list = [1]
        # metrics we want to keep track of
        populations = list()
        free_energy = dict()
        average_cost = list()
        best = list()
        walker_z = list()
        walker_pos, new_walker_pos = dict(), dict()
        
        taus_over_time = {i:0 for i in range(self.walkers_t1)}
        # generate a state of random walkers with their costs, need to change such that we are generating tours instead of other stuff.

        # generate a state of random walkers with their costs 
        walker_pos = {i:[[random.uniform(self.i1[0][0], self.i1[0][-1]),
                             random.uniform(self.i2[0][0], self.i2[0][-1])]] for i in range(self.walkers_t1)}
        # append the cost of each state 
        for k,v in walker_pos.items():
            walker_pos[k].append(self.f(walker_pos[k][0]))

        # gets the maximum value of the key 
        max_key = self.max_key(walker_pos)
        best_cost = [[1, walker_pos[max_key][0], walker_pos[max_key][1]]]

        for temp_step in range(2, self.Tmax+2):            
            # calculate the temperature from temp step 2 onward

            fraction = 1/temp_step

            if temp_step > 2:
                if self.lams == 0:
                    T = self.multiplier * fraction if self.multiplier < 1 & temp_step > 2 else fraction
            else:
                T = fraction 
            temp = int(np.round(1/T))

            T_list.append(temp)
            populations.append(self.walkers_t1)
            
            params = (T_list[-2], T_list[-1], np.mean(populations))

            stat_weight_ratio, partition_function, Q, tau = self.partition_calc(walker_pos, *params)
            new_params = [walker_pos] + [params[-1]] + [stat_weight_ratio, Q, tau]
            resampled_walker = self.resample_population(*new_params)

            #taus_over_time = {k:(taus_over_time[k]+v) for k,v in tau.items()}

            # explore a new city configuration for each walker (the annealing step)
            for k,v in walker_pos.items():
                
                #########################################
                ### This is where PA differes from SA ###
                #########################################

                if resampled_walker[k] > 0:

                    new_walker = list(new_walker_pos.keys())[-1]+1 if len(list(new_walker_pos.keys())) != 0 else 0

                    for i in range(resampled_walker[k]):
                        new_walker_pos[new_walker+i] = walker_pos[k][:]
                        #walker_index[k].append(new_walker + i)

                costs = round(walker_pos[k][1], 2)
                states = walker_pos[k][0]

                # if costs not in self.energy_landscape.keys():
                #     self.energy_landscape[costs] = 1
                # else:
                #     self.energy_landscape[costs] = self.energy_landscape[costs] + 1

                walker_pos_check = walker_pos.copy()
                for step in range(self.exploration_space):
                    new_state = self.random_neighbour(states)
                    new_cost = self.f(new_state)
                    # walker_pos_check[k][1] = new_cost
                    # new_stat_weight_ratio, new_partition_function, new_Q, new_tau = self.partition_calc(walker_pos_check, *params)
                    # walker_z.append([temp_step, step, k, new_partition_function])
                    
                if new_cost < costs or self.acceptance_probability(costs,
                      new_cost, T) >= random.uniform(0, 1):
                    states, costs = new_state, new_cost
                
                if self.lams == 1:
                    self.acceprate = 1 / 500 * (499 * self.acceptrate + 1)
                else:

                    if self.lams == 1:

                        self.acceptrate = 1 / 500 * (499 * self.acceptrate)

                        # check conditions

                        if fraction < 0.15:
                            LamRate = 0.44 + 0.56 * 560 ** (-temp_step
                              / (self.Tmax * 0.15))
                        elif fraction < 0.65:
                            LamRate = 0.44
                        else:
                            LamRate = 0.44 * 440 ** ((-fraction - 0.65) / 0.35)

                        if LamRate < self.acceptrate:
                          T *= 0.99
                        else:
                          T *= 1 / 0.999

                walker_pos[k][0], walker_pos[k][1] = states, costs

                # reassign to best cost if greater than the current best cost
                if costs < best_cost[-1][2]:
                    best_cost.append([temp_step, states, costs/self.initial]) # should i be putting the state or the walker? none of them are meaningful anyway... 

            best.append(best_cost[-1][2])
                    
            all_costs = np.array([walker_pos[k][1] for k,v in walker_pos.items()])
            
            average_cost.append(np.mean(all_costs))

            free_energy[temp_step] = math.log(Q) + math.log(self.walkers_t1)
            # print(free_energy)
            self.check_correct(all_costs/self.initial)
            
            walker_pos = new_walker_pos.copy()
            self.walkers_t1 = self.walkers_t2
            self.walkers_t2 = len(walker_pos)
            new_walker_pos = dict()


        return (
                self.energy_landscape,
                average_cost,
                self.cumulative_correct,
                free_energy,
                best_cost,
                best,
                populations,
                T_list, 
                walker_z,
                taus_over_time
                )


if __name__ == '__main__':
	
	# explore represents the number of walker: 50, 100, 200, 300, 400, 500
    explore = [50] + [i for i in range(100, 501, 100)] 
    configs = [[3000, 30, 10e-2, 0, 1, 0.5, 0, "multinomial"], [3000, 30, 10e-2, 0, 0.8, 0.5, 0, "multinomial"], [3000, 30, 10e-2, 0, 1, 0.5, 1, "multinomial"], [3000, 30, 10e-2, 0, 1, 0.5, 0, "poisson"], [3000, 30, 10e-2, 0, 0.8, 0.5, 0, "poisson"], [3000, 30, 10e-2, 0, 1, 0.5, 1, "poisson"]]
    iters = 100

    # we are exploring 3 different configurations for poisson and multinomial each: total = 6
    for config in configs:
        print("Current config: {}".format(config))
        sys.stdout.flush()

        
        choice = config[-1]
        multiplier = config[5]
        lams = config[-2]

        for k in explore:
            print("Current number of walkers: {}".format(k))
            sys.stdout.flush()

            config = config[0:2] + [k] + config[2:]

            pop_anneal = {"choice":list(), "multiplier":list(), "lams": list(), "walkers":list(), "run":list(), 'temperature':list(), \
                      "converged_perc":list(), "best_cost":list(), "avg_cost_temp": list(), "temp_pop":list()}
            start_time = time.time()

            for i in range(iters):
                start_time2 = time.time()

                print("Current Iteration: {}".format(i))
                sys.stdout.flush()

                a = PAAnneal(*config)
                energy_landscape, average_cost, cumulative, free_energy, best_cost, all_best, population, temp, walker_z, taus_over_time = a.anneal()
    			
                #temp = [0] + temp
    			#temp = temp[:-2]
                total_population = np.sum(population)
    			#new_divergence = np.abs([0 if math.isinf(v) == True else v for k,v in kl_divergence.items()])
                temp = temp[1::]

                
                
                for j,t in enumerate(temp):
                    pop_anneal['choice'].append(choice)
                    pop_anneal['multiplier'].append(multiplier)
                    pop_anneal['lams'].append(lams)
                    pop_anneal['walkers'].append(k)
                    pop_anneal['run'].append(i)
                    pop_anneal['temperature'].append(t)
                    #pop_anneal['free_energy'].append(free_energy[j])
                    pop_anneal["converged_perc"].append(cumulative/total_population)
                    pop_anneal["best_cost"].append(best_cost[-1][2])
                    pop_anneal["avg_cost_temp"].append(average_cost[j])
                    pop_anneal["temp_pop"].append(population[j])
				
                for key in list(pop_anneal.keys()):
                    print("{}:{}".format(key, len(pop_anneal[key])))

                print("Loop Iteration Execution Time: --- %s seconds ---" % (time.time() - start_time2))

            anneal_run = pd.DataFrame.from_dict(pop_anneal)
    		#anneal_run.head()
            anneal_run.to_csv("PA_iter{}_run_lams{}_{}_{}_test.csv".format(i, lams, multiplier, k))

            print("Configuration Iteration Execution Time: --- %s seconds ---" % (time.time() - start_time))