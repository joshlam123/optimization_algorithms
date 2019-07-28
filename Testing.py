import math
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../Simulated Annealing/Discrete')
sys.path.insert(0, '../Population Annealing/Discrete')
import random
from multiprocessing import Pool
import os
from operator import itemgetter
import time
import logging
import time

from gen_cities import GenCities
from discrete_sa_anneal import SAAnneal
from discrete_pa_anneal import PAAnneal
from greedy import GreedyTSP

logging.basicConfig(filename='testing_app.log', filemode='w', format = '%(asctime)s  %(levelname)-10s %(processName)s  %(name)s %(message)s')


# def spawn_cities(no_cities, **args):
#     create_city = GenCities(no_cities)
#           # generate the random cities
#     create_city.generate_cities()
#     create_city.generate_initial_tour() 
#     create_city.precomp_distances()

#     return create_city


if __name__ == '__main__':

    cities = 100
    explore = 50
    batch_iter = 5

    start_cities = list()

    batch = sys.argv[1]

    try:
        for i in range(batch_iter):
            print("Current Batch Iteration: {}".format(i))
            sys.stdout.flush()

            # generate new city in batch
            anneal_pairs = {"cities":list(), "lowest":list(), "sim_anneal_best":list(), "sim_anneal_solved":list(), \
                    "pop_anneal_best":list(), "pop_anneal_solved":list(), "free_energy":list(), "pop_population":list()}

            global Ncity
            start = time.time()
            Ncity = GenCities(no_cities=cities)
            towns = Ncity.generate_cities()
            table = Ncity.precomp_distances()
            print('Time taken to Precompute Distances: {}'.format(time.time() - start))
            sys.stdout.flush()

            a = GreedyTSP(Ncity.cities, Ncity.start_city, Ncity.table_distances)
            total_distance, lowest = a.greedy_this()


            start_time = time.time()
            print("Current number of walkers: {}".format(explore))                
            sys.stdout.flush()

             # in this case the lowest is the average number of cities

            # lowest_distance = list()
            # run 100 iterations ONCE for each set of cities

            # generate the greedy lowest number
            # for i in range(100):
            #     global Qcity
            #     Qcity = GenCities(no_cities=j)
            #     #start_cities.append(Qcity.start_city)
            #     a = GreedyTSP(Qcity.cities, Qcity.start_city, Qcity.table_distances)
            #     total_distance, average_distance = a.greedy_this()
            #     lowest_distance.append(total_distance)

            # lowest = np.mean(np.array(lowest_distance))
            
            sa = SAAnneal(Ncity, maxsteps=101, multiplier=0.8, swaps=round((Ncity.n)**0.5), correct=lowest)
            best_tour, current_tour, best_tour_total, current_tour_total, sa_temp, tours, sa_cum = sa.anneal()
            pa = PAAnneal(Ncity, maxsteps=101, multiplier=0.8, swaps=round((Ncity.n)**0.5), walkers=explore, correct=lowest)
            energy_landscape, average_cost, pa_cum, free_energy, best_cost_replace, best_cost, population, pa_temp = pa.anneal()

            free_energies = np.mean([free_energy[k] for k,v in free_energy.items()])

            print("Batch Iteration Execution Time: --- %s seconds ---" % (time.time() - start_time))
            sys.stdout.flush()


            sim_anneal_min = min(best_tour)
            pop_anneal_min = best_cost[2]

            # sim_anneal_best = np.sum(best_tour) 
            # # sim_anneal_cum = np.sum(sim_anneal['cumulative']) / iters
            # pop_anneal_best = np.sum(best_cost) 
            # pop_anneal_cum = np.sum(pop_anneal['cumulative']) / np.sum(pop_anneal['population']) 

            for k in range(len(sa_temp)):
                anneal_pairs['cities'].append(cities)
                anneal_pairs['lowest'].append(lowest)
                anneal_pairs['sim_anneal_best'].append(best_tour[k])
                anneal_pairs['pop_anneal_best'].append(best_cost[k])

                pa_correct = 1 if pop_anneal_min <= lowest else 0
                sa_correct = 1 if sim_anneal_min <= lowest else 0
                print("SA Correct:{}\nPA Correct:{}".format(sa_correct,pa_correct))
                anneal_pairs['sim_anneal_solved'].append(sa_correct)
                anneal_pairs['pop_anneal_solved'].append(pa_correct)

                anneal_pairs['free_energy'].append(free_energy[k])
                anneal_pairs['pop_population'].append(population[k])

            df = pd.DataFrame.from_dict(anneal_pairs)
            df.to_csv("anneal_compare{}_batch{}_iter{}.csv".format(explore, batch, i))

            print("Configuration Execution Time: --- %s seconds ---" % (time.time() - start_time))
            sys.stdout.flush()

    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

