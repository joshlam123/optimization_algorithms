import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import random
from gen_cities import GenCities
# Display all floats rounded off to 1 decimal place
pd.options.display.float_format = '{:,.1f}'.format

sns.set()
# Width = 16, Height = 6
DIMS=(16, 6)
import os


class PAAnneal:

    '''
    Pass the max steps you want to take to the annealer function
    '''

    def __init__(
          self,
        super_energies,
        func,
        mesh,  
        Ncity,
        maxsteps=500,
        multiplier=1,
        control_t=1,
        swaps=round((50)**0.5),
        explore=30,
        walkers=10,
        error_thres=10e-2, 
        correct=0.0,
        choice='multinomial'
      #accs = [500, 1, 1, 0.5, 0, round((Ncity.n)**0.5), 30]
          ):
        '''
        inputs: total number of steps to try, geometric multiplier for annealing schedule
        Initialize parameters
        output: none
        '''
        self.Ncity = Ncity
        self.cities, self.start_city, self.table_distances = self.Ncity.cities, self.Ncity.start_city, self.Ncity.table_distances
        self.correct_answer, self.error_threshold, self.cumulative_correct = correct, error_thres, 0.0
        self.threshold, self.control = multiplier, control_t
        self.Tmax, self.exploration_space, self.swaps = maxsteps, explore, swaps
        self.lams, self.acceptrate = 0, 0.5  

          # need to change the walkers to match discrete case
        self.walkers_t1, self.walkers_t2, self.initial = walkers, walkers, walkers
        self.walker_pos, self.new_walker_pos = dict(), dict()

        self.energy_landscape = dict()

          # e_diff is a lambda function used to calculate the ratio of statistical weight
        self.e_diff = lambda x, y: np.exp(-(x[1] - x[0]) * y) 
        self.distance = lambda x, y: np.sqrt((x[0] - y[0]) ** 2 + (x[1]
                      - y[1]) ** 2)
    
    def resample_population(self, walker_pos, mean_val, stat_weight_ratio, Q, tau, choice='poisson'):
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


    def max_key(self):
        '''
        inputs: none
        finds the minimum value in the dictionary of walkers
        outputs: key of the lowest (best) cost value in the entire dictionary of walkers
        '''
        return min(self.walker_pos.keys(), key=(lambda k: self.walker_pos[k][1]))      

    
    def f(self, tour):
        '''
        input: tour (list)

        Function that evaluates the cost of a given x1, x2 (euclidean distance)

        output: single cost
        '''
        distances = [self.table_distances[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)]

        total_distance = sum(distances)

        return total_distance

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
        idx = range(self.Ncity.n - 1)
        for i in range(self.swaps):
            (i1, i2) = random.sample(idx, 2)
            (tour[i1], tour[i2]) = (tour[i2], tour[i1])

        tour = [self.start_city] + tour + [self.start_city]
        cost = self.f(tour)
        return (tour, cost)
    
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

        # params related to returning the cost and deviation from the optimal objective function
        # deviation = list()

        # params related to Lam's Annealing Schedule

        T_list = [1]
        # metrics we want to keep track of
        populations = list()
        free_energy = dict()
        average_cost = list()
        best = list()
        walker_z = list()
        walker_pos, new_walker_pos = dict(), dict()
        resampled_B, resampled_B_prime = dict(), dict()
        configs_explored = dict()
        taus_over_time = dict()

        # generate a state of random walkers with their costs, need to change such that we are generating tours instead of other stuff.

        # something here that is causing it not to generate properly
        for i in range(self.walkers_t1):
            tour = self.Ncity.generate_initial_tour()
            walker_pos[i] = [tour, self.f(tour)]

        max_key = self.max_key(walker_pos)
        best_tour = [[1, walker_pos[max_key][0], walker_pos[max_key][1]/10]]

        for temp_step in range(2, self.Tmax+2):            
            # calculate the temperature from temp step 2 onward

            fraction = 1/temp_step

            if temp_step > 2:
                if self.lams == 0:
                    T = self.threshold * fraction if self.threshold < 1 else fraction
            else:
                T = fraction 

            T_list.append(int(np.round(1/T)))
            populations.append(self.walkers_t1)
            
            params = (T_list[-2], T_list[-1], np.mean(populations))

            stat_weight_ratio, partition_function, Q, tau = self.partition_calc(walker_pos, *params)
            new_params = [walker_pos] + [params[-1]] + [stat_weight_ratio, Q, tau]
            resampled_walker = self.resample_population(*new_params)

            # explore a new city configuration for each walker (the annealing step)
            for k,v in walker_pos.items():

                if resampled_walker[k] > 0:
                    new_walker = list(new_walker_pos.keys())[-1] if len(list(new_walker_pos.keys())) != 0 else 0

                    for i in range(resampled_walker[k]):
                        new_walker_pos[new_walker+i] = walker_pos[k][:]

                costs = round(walker_pos[k][1], 2)
                states = walker_pos[k][0]

                if costs not in self.energy_landscape.keys():
                    self.energy_landscape[costs] = 1
                else:
                    self.energy_landscape[costs] = self.energy_landscape[costs] + 1

                walker_pos_check = walker_pos.copy()
                for step in range(self.exploration_space):
                    (new_tour, new_cost) = self.swap_random(states)
                    
                    # walker_pos_check[k][1] = new_cost
                    # stat_weight_ratio, new_partition_function, new_Q, tau = self.partition_calc(walker_pos_check, *params)
                    # walker_z.append([temp_step, step, k, new_partition_function])
                    
                if new_cost < costs or self.acceptance_probability(costs,
                      new_cost, T) >= random.uniform(0, 1):
                    states, costs = new_tour, new_cost

                walker_pos[k][0], walker_pos[k][1] = states, costs


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

                # reassign to best cost if greater than the current best cost
                if costs < best_tour[-1][2]:
                    best_tour.append([temp_step, states, costs/self.initial]) # should i be putting the state or the walker? none of them are meaningful anyway... 

            best.append(best_tour[-1][2])
                    
            all_costs = np.array([walker_pos[k][1] for k,v in walker_pos.items()])
            
            average_cost.append(np.mean(all_costs))
    
            free_energy[temp_step] = math.log(Q) + math.log(self.walkers_t1)
    
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
                best_tour,
                best,
                populations,
                T_list, 
                walker_z
                )


