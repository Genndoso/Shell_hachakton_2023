import os
import pandas as pd
from src.commons.utils import get_depot_param_space, get_refinery_param_space
from src.supply_chain_opt.greedy_algorithm import Greedy_algorithm
from skopt import gp_minimize

os.chdir('/Users/user/PycharmProjects/Shell_hachakton_2023')
data = pd.read_csv('data/Biomass_History.csv', index_col=0)
dist = pd.read_csv('data/Distance_Matrix.csv', index_col=0)
biomass_2010 = data['2010']

# Greedy Algorithm
GA = Greedy_algorithm(biomass = biomass_2010,
                      dist = dist)

# Get space for depot optimization
space_depot, number_of_depots = get_depot_param_space(biomass=biomass_2010)
GA.number_of_depots = number_of_depots

# Optimize depot locations
GA.optimize = True
res_depot = gp_minimize(GA.objective_depot, space_depot, n_calls=100)
GA.optimize = False
depot_cost, _, biomass_demand_supply_solution = GA.objective_depot(space=res_depot.x)

# Get space for refinery optimization
space_refinery, number_of_refineries = get_refinery_param_space(biomass=GA.depot_biomass_supply)
GA.number_of_refineries = number_of_refineries

# Optimize refinery locations
GA.optimize = True
res_refinery = gp_minimize(GA.objective_refinery, space_refinery, n_calls=100)
GA.optimize = False
refinery_cost, _, pellet_demand_supply_solution = GA.objective_refinery(space=res_refinery.x)

print(f'Total cost is {depot_cost + refinery_cost}. '
      f'Depot transport cost: {depot_cost}. '
      f'Refinery transport cost: {refinery_cost}.')

