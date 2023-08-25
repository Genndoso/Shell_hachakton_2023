import pandas as pd
from pyscipopt import Model, quicksum, multidict
import os
os.chdir('/Users/user/PycharmProjects/Shell_hachakton_2023')
import warnings
warnings.filterwarnings("ignore")
from src.supply_chain_opt.CFLP import preprocessing, predict_biomass, CFLP_recalculate_routes,\
    get_prediction, get_locations, get_routes_solution, cluster_preprocessing

data = pd.read_csv('data/Biomass_History.csv', index_col=0)
dist = pd.read_csv('data/Distance_Matrix.csv', index_col=0)
#data = data.sample(200)
#indices_list = list(data.index)
#dist = dist.loc[data.index, [str(i) for i in indices_list]]

EPS = 1.e-6
year1 = '2018'; year2 = '2019'
depot_cap = 20000
refinery_cap = 100000
biomass_percentage_to_collect = 0.85
num_of_depot_guesses = 5


def kmedian(I, J, c, k):
    model = Model("k-median")
    x, y = {}, {}
    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in I:
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    for i in I:
        model.addCons(quicksum(x[i, j] for j in J) == 1, "Assign(%s)" % i)
        for j in J:
            model.addCons(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))
    model.addCons(quicksum(y[j] for j in J) == k, "Facilities")
    model.setObjective(quicksum(c[i, j] * x[i, j] for i in I for j in J), "minimize")
    model.data = x, y

    return model


if __name__ == '__main__':

    # Prediction
    prediction = predict_biomass(data)
    # Preprocess data
    indices = preprocessing(prediction, dist, biomass_percentage=biomass_percentage_to_collect)
    prediction = prediction.loc[indices]
    average_biomass = prediction.iloc[:, -2:].mean(axis=1)
    dist = dist.loc[indices, [str(i) for i in indices]]

    # Number of depots and refineries based on biomass demand
    number_of_depots = max(int(prediction['2018'].sum() / depot_cap) + 1, int(prediction['2019'].sum() / depot_cap) + 1)
    number_of_refineries = max(int(prediction['2018'].sum() / refinery_cap) + 1, int(prediction['2019'].sum() / refinery_cap) + 1)
    print(f'number of depots is {number_of_depots} ')
    print(f'number of refineries is {number_of_refineries} ')
    # Site locations and biomass demand. Site locations taken from prediction df
    site_locations, biomass_demand_year1 = multidict(dict(prediction[year1]))
    _, biomass_demand_year2 = multidict(dict(prediction[year2]))
    # Get an initial guess on possible_depot_location
    _, _, possible_depot_locations = cluster_preprocessing(data=prediction, num_clusters=number_of_depots, num_closest_points=num_of_depot_guesses)
    possible_depot_locations = list(possible_depot_locations)
    # Possible depot locations after initial guess on possible depot locations
    capacity1 = pd.Series(depot_cap, index=possible_depot_locations)  # assuming all possible locations
    _, depot_capacity = multidict(dict(capacity1))

    # Possible refinery locations taken from all locations in prediciton df
    capacity2 = pd.Series(refinery_cap, index=prediction[year1].index)
    possible_refinery_locations, refinery_capacity = multidict(dict(capacity2))
    # Dist matrix. Use .loc to take str column names from dist
    site_depot_dist = dist.loc[site_locations, [str(pdl) for pdl in possible_depot_locations]].reset_index().melt(id_vars='index')

    site_depot_dist['variable'] = site_depot_dist['variable'].astype('int')
    site_depot_dist = {(row[0], row[1]): row[2] for row in site_depot_dist.values}
    # Weighted dist matrix
    weighted_dist = (dist.loc[site_locations, [str(pdl) for pdl in possible_depot_locations]].T * average_biomass).T
    site_depot_weighted_dist = weighted_dist.reset_index().melt(id_vars='index')
    site_depot_weighted_dist['variable'] = site_depot_weighted_dist['variable'].astype('int')
    site_depot_weighted_dist = {(row[0], row[1]): row[2] for row in site_depot_weighted_dist.values}


    ''' FIRST FIND OPTIMAL LOCATIONS. BASED ON THE PREDEFINED NUMBER OF FACILITIES AND DIST MATRIX.'''
    model1 = kmedian(I=site_locations,
                     J=possible_depot_locations,
                     c=site_depot_weighted_dist,
                     k=number_of_depots)
    model1.optimize()
    x1, y1 = model1.data
    depot_locations = [j for j in y1 if model1.getVal(y1[j]) > EPS]
    boolean_depot_locations = {j: abs(model1.getVal(y1[j])) for j in possible_depot_locations}

    # Depot - Refinery distance matrix
    depot_refinery_dist = dist.loc[depot_locations, [str(prl) for prl in possible_refinery_locations]]
    depot_refinery_dist = depot_refinery_dist.reset_index().melt(id_vars='index')
    depot_refinery_dist['variable'] = depot_refinery_dist['variable'].astype('int')
    depot_refinery_dist = {(row[0], row[1]): row[2] for row in depot_refinery_dist.values}

    model2 = kmedian(I=depot_locations,
                     J=possible_refinery_locations,
                     c=depot_refinery_dist,
                     k=number_of_refineries)
    model2.optimize()
    x2, y2 = model2.data
    refinery_locations = [j for j in y2 if model2.getVal(y2[j]) > EPS]
    boolean_refinery_locations = {j: abs(model2.getVal(y2[j])) for j in possible_refinery_locations}

    print(f'Depot locations: {depot_locations}')
    print(f'Refinery locations: {refinery_locations}')


    ''' OPTIMIZE SUPPLY CHAIN ROUTES FOR EACH YEAR '''
    # 2018 SITE - DEPOT ROUTES
    model3 = CFLP_recalculate_routes(I=site_locations,
                                     J=possible_depot_locations,
                                     d=biomass_demand_year1,
                                     M=depot_capacity,
                                     c=site_depot_dist,
                                     y=boolean_depot_locations)
    model3.optimize()
    x3 = model3.data
    depot_routes_year1 = [(i, j) for (i, j) in x3 if model3.getVal(x3[i, j]) > EPS]

    # Get pellet demand at each depot
    pellet_demand_year1 = {dl: sum([model3.getVal(x3[sl, dl]) for sl in site_locations if model3.getVal(x3[sl, dl]) > EPS])
                            for dl in depot_locations}


    # 2018 DEPOT - REFINERY ROUTES
    model4 = CFLP_recalculate_routes(I=depot_locations,
                                     J=possible_refinery_locations,
                                     d=pellet_demand_year1,
                                     M=refinery_capacity,
                                     c=depot_refinery_dist,
                                     y=boolean_refinery_locations)
    model4.optimize()
    x4 = model4.data
    refinery_routes_year1 = [(i, j) for (i, j) in x4 if model4.getVal(x4[i, j]) > EPS]


    # 2019 SITE - DEPOT ROUTES
    model5 = CFLP_recalculate_routes(I=site_locations,
                                     J=possible_depot_locations,
                                     d=biomass_demand_year2,
                                     M=depot_capacity,
                                     c=site_depot_dist,
                                     y=boolean_depot_locations)
    model5.optimize()
    x5 = model5.data
    depot_routes_year2 = [(i, j) for (i, j) in x5 if model5.getVal(x5[i, j]) > EPS]

    # Get pellet demand at each depot
    pellet_demand_year2 = {dl: sum([model5.getVal(x5[sl, dl]) for sl in site_locations if model5.getVal(x5[sl, dl]) > EPS])
                            for dl in depot_locations}


    # 2019 DEPOT - REFINERY ROUTES
    model6 = CFLP_recalculate_routes(I=depot_locations,
                                     J=possible_refinery_locations,
                                     d=pellet_demand_year2,
                                     M=refinery_capacity,
                                     c=depot_refinery_dist,
                                     y=boolean_refinery_locations)
    model6.optimize()
    x6 = model6.data
    refinery_routes_year2 = [(i, j) for (i, j) in x6 if model6.getVal(x6[i, j]) > EPS]

    print(f"Optimal depot transport&utilization cost {year1} = ", model3.getObjVal())
    print(f"Site-Depot routes for {year1} (from,to): ", depot_routes_year1)
    print(f"Optimal refinery transport&utilization cost {year1} = ", model4.getObjVal())
    print(f"Depot-Refinery routes for {year1} (from,to): ", refinery_routes_year1)
    print(f"Optimal depot transport&utilization cost {year2} = ", model5.getObjVal())
    print(f"Site-Depot routes for {year2} (from,to): ", depot_routes_year2)
    print(f"Optimal refinery transport&utilization cost {year2} = ", model6.getObjVal())
    print(f"Depot-Refinery routes for {year2} (from,to): ", refinery_routes_year2)
    print(f'Transport & Underutilizaiton cost = {model3.getObjVal()+model4.getObjVal()+model5.getObjVal()+model6.getObjVal()}')


    ''' COMPILE SUBMISSION FILE'''
    depot_locations_solution = get_locations(facilities=depot_locations,
                                             year='20182019',
                                             data_type='depot_location')
    refinery_locations_solution = get_locations(facilities=refinery_locations,
                                                year='20182019',
                                                data_type='refinery_location')
    biomass_prediction_year1_solution = get_prediction(data=prediction, year=year1)
    biomass_prediction_year2_solution = get_prediction(data=prediction, year=year2)
    depot_routes_year1_solution = get_routes_solution(model=model3,
                                                      x=x3,
                                                      routes=depot_routes_year1,
                                                      year=year1,
                                                      data_type='biomass_demand_supply')
    refinery_routes_year1_solution = get_routes_solution(model=model4,
                                                         x=x4,
                                                         routes=refinery_routes_year1,
                                                         year=year1,
                                                         data_type='pellet_demand_supply')
    depot_routes_year2_solution = get_routes_solution(model=model5,
                                                      x=x5,
                                                      routes=depot_routes_year2,
                                                      year=year2,
                                                      data_type='biomass_demand_supply')
    refinery_routes_year2_solution = get_routes_solution(model=model6,
                                                         x=x6,
                                                         routes=refinery_routes_year2,
                                                         year=year2,
                                                         data_type='pellet_demand_supply')

    solution = pd.concat([depot_locations_solution,
                          refinery_locations_solution,
                          biomass_prediction_year1_solution,
                          depot_routes_year1_solution,
                          refinery_routes_year1_solution,
                          biomass_prediction_year2_solution,
                          depot_routes_year2_solution,
                          refinery_routes_year2_solution], axis=0, ignore_index=True)
    solution.to_csv('submission/SCIP_solution.csv')