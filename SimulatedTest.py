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
import logging

from gen_cities import GenCities
from sa_anneal import SAAnneal
import cProfile

def run_function(a):
    cProfile.run('a.anneal()')
    best_tour, current_tour, best_tour_total, current_tour_total, temp, tours, cum = a.anneal()
    return best_tour, current_tour, best_tour_total, current_tour_total, temp, tours

def get_df(best_tour, current_tour, best_tour_total, current_tour_total, temp, tours):
    tries_temp = {"run":list(), "temp":list(), "best_tour_avg":list(), "best_tour_total":list(), "current_tour_avg":list(), "current_tour_total":list()}
    cost_keys = len(list(best_tour))
    for k in range(cost_keys):
        tries_temp['run'].append(k)
        tries_temp['temp'].append(temp[k])
        tries_temp['best_tour_avg'].append(best_tour[k])
        tries_temp['best_tour_total'].append(best_tour_total[k])
        tries_temp['current_tour_avg'].append(current_tour[k])
        tries_temp['current_tour_total'].append(current_tour_total[k])
          #tries_temp['deviations'].append(deviations[k][-1][-1])

    temp_df = pd.DataFrame.from_dict(tries_temp)
    temp_mean = temp_df.groupby(['temp']).mean().reset_index()
    return temp_mean[temp_mean['current_tour_avg']==temp_mean['current_tour_avg'].min()]['current_tour_avg']

# first: generate cities
cities = [20, 50, 100, 300, 500]
no_of_steps = [5000, 30000, 100000]

options = [[1, 1, 0], [0.8, 0, 0], [1,1,1]]

current_tour_avg = list()

for i in cities:
    Ncity = GenCities(no_cities=i)
    for j in no_of_steps:
        for k in options:
            print("{}:{}".format(i,j))
            params = [Ncity, j] + k + [round((Ncity.n)**0.5)]
            a = SAAnneal(*params)
            best_tour, current_tour, best_tour_total, current_tour_total, temp, tours = run_function(a) 
            lowest = get_df(best_tour, current_tour, best_tour_total, current_tour_total, temp, tours)
            current_tour_avg.append(i, params[2], params[4], j, params[5], lowest)

df = pd.DataFrame.from_records(current_tour_avg).T
df.to_csv("Simulated Annealing TSP Lowest Cost.csv")
