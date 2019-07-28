from tsp_anneal import TSPAnneal
from brute import BruteTSP
from greedy import GreedyTSP
from gen_cities import GenCities
import pandas as pd

def get_cities():
    n = input('Please let us know how many cities you would like for your TSP map: \n')
    return int(n)

def spawn_cities(no_cities, **args):
    create_city = GenCities(no_cities)
          # generate the random cities
    create_city.generate_cities()
    create_city.generate_initial_tour() 
    create_city.precomp_distances()
    cities = create_city.cities
    start_city = create_city.start_city
    initial_tour = create_city.init_tour
    print("cities:{},\n start city:{},\n initial tour: {}".format(cities, start_city, initial_tour))
    #pd.DataFrame(initial_tour).rename(columns={0:"tour_order"}).to_csv("initial_tour.csv")
    return create_city


if __name__ == '__main__':
    n = get_cities()
    #Ncity = spawn_cities(no_cities=30)
    Ncity = spawn_cities(no_cities=n)

#     # sample way to run the code
#     ## anneal
#     tries_temp = {"temp":list(), "avg_tour_len":list(), "cost":list()}#, "deviations":list()}


#     a = TSP_Anneal(n_city=Ncity, maxsteps=5000, control_t=1, lams=1, swaps=1)
#     tours, avg_len = a.anneal()
#     cost_keys = list(avg_len.keys())
#     for k in cost_keys:
#           tries_temp['temp'].append(k)
#           tries_temp['avg_tour_len'].append(avg_len[k])
#           tries_temp['cost'].append(tours[k])
#           #tries_temp['deviations'].append(deviations[k][-1][-1])

#     temp_df = pd.DataFrame.from_dict(tries_temp)
#     temp_mean = temp_df.groupby(['temp']).mean().reset_index()
#     temp_mean[temp_mean['avg_tour_len'] == temp_mean['avg_tour_len'].min()]
#     #temp_mean.to_csv("tsp_anneal_100.csv")

#     ## greedy
#     tries_greedy = {"avg_tour_len":list(), "cost":list()}#, "deviations":list()}

#     greedy = GreedyTSP(n_city=Ncity)
#     total_distance, average_distance = greedy.greedy_this()
#     tries_greedy['avg_tour_len'].append(average_distance)
#     tries_greedy['cost'].append(total_distance)
#     #   print("The total distance from Greedy TSP is: {}".format(total_distance))
#     #   print("The average distance from Greedy TSP is: {}".format(average_distance))

#     greedy_df = pd.DataFrame.from_dict(tries_temp)
#     greedy_mean = greedy_df.groupby(['temp']).mean().reset_index()
#     greedy_mean[greedy_mean['avg_tour_len'] == greedy_mean['avg_tour_len'].min()]




    #######################################################
    ########## RUN BRUTE FORCE AT YOUR OWN RISK ###########
    ########## CODE COMMENTED OUT FOR YOUR SAFETY #########
    #######################################################

    # 20 CITIES = 20! = 2 X 10**18

    # bruteforce = BruteTSP(n_city=Ncity)
    # tries_brute = {"avg_tour_len":list(), "cost":list()}#, "deviations":list()}
    # total_distance, average_distance, perms = bruteforce.brute_this()
    # tries_brute['avg_tour_len'].append(average_distance)
    # tries_brute['cost'].append(total_distance)
    # #   print("The total distance from Greedy TSP is: {}".format(total_distance))
    # #   print("The average distance from Greedy TSP is: {}".format(average_distance))

    # brute_df = pd.DataFrame.from_dict(tries_brute)
    # brute_mean = brute_df.groupby(['temp']).mean().reset_index()
    # brute_mean[brute_mean['avg_tour_len'] == greedy_mean['avg_tour_len'].min()]