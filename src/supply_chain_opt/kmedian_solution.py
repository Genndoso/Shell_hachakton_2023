import pandas as pd
from pyscipopt import Model, quicksum, multidict
import os
os.chdir('/Users/user/PycharmProjects/Shell_hachakton_2023')
from src.supply_chain_opt.CFLP import predict_biomass, CFLP_recalculate_routes,\
    get_prediction, get_locations, get_routes_solution

data = pd.read_csv('data/Biomass_History.csv', index_col=0)
dist = pd.read_csv('data/Distance_Matrix.csv', index_col=0)

EPS = 1.e-6
year1 = '2018'; year2 = '2019'
depot_cap = 20000
refinery_cap = 100000


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
    average_biomass = prediction.iloc[:, -2:].mean(axis=1)
    # Number of depots and refineries based on biomass demand
    number_of_depots = max(int(prediction['2018'].sum() / depot_cap) + 1, int(prediction['2019'].sum() / depot_cap) + 1)
    number_of_refineries = max(int(prediction['2018'].sum() / refinery_cap) + 1, int(prediction['2019'].sum() / refinery_cap) + 1)
    print(f'number of depots is {number_of_depots} ')
    print(f'number of refineries is {number_of_refineries} ')
    # Site locations and biomass demand
    site_locations, biomass_demand_year1 = multidict(dict(prediction[year1]))
    _, biomass_demand_year2 = multidict(dict(prediction[year2]))
    # Possible depot and refinery locations
    capacity1 = pd.Series(depot_cap, index=prediction[year1].index)  # assuming all possible locations
    possible_depot_locations, depot_capacity = multidict(dict(capacity1))
    capacity2 = pd.Series(refinery_cap, index=prediction[year1].index)
    possible_refinery_locations, refinery_capacity = multidict(dict(capacity2))
    # Dist matrix
    site_depot_dist = dist.iloc[site_locations, possible_depot_locations].reset_index().melt(id_vars='index')
    site_depot_dist['variable'] = site_depot_dist['variable'].astype('int')
    site_depot_dist = {(row[0], row[1]): row[2] for row in site_depot_dist.values}
    # Weighted dist matrix
    weighted_dist = (dist.iloc[site_locations, possible_depot_locations].T * average_biomass).T
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
    boolean_depot_locations = {j: model1.getVal(y1[j]) for j in possible_depot_locations}
    print("Depots at nodes: ", depot_locations)

    # Depot - Refinery distance matrix
    depot_refinery_dist = dist.iloc[depot_locations, possible_refinery_locations].reset_index().melt(id_vars='index')
    depot_refinery_dist['variable'] = depot_refinery_dist['variable'].astype('int')
    depot_refinery_dist = {(row[0], row[1]): row[2] for row in depot_refinery_dist.values}

    model2 = kmedian(I=depot_locations,
                     J=possible_refinery_locations,
                     c=depot_refinery_dist,
                     k=number_of_refineries)
    model2.optimize()
    x2, y2 = model1.data
    refinery_locations = [j for j in y2 if model2.getVal(y2[j]) > EPS]
    boolean_refinery_locations = {j: model2.getVal(y2[j]) for j in possible_refinery_locations}
    print("Refineries at nodes: ", refinery_locations)


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
    print(f"Optimal depot transport&utilization cost {year1} = ", model3.getObjVal())
    print(f"Site-Depot routes for {year1} (from,to): ", depot_routes_year1)
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
    print(f"Optimal refinery transport&utilization cost {year1} = ", model4.getObjVal())
    print(f"Depot-Refinery routes for {year1} (from,to): ", refinery_routes_year1)

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
    print(f"Optimal depot transport&utilization cost {year2} = ", model5.getObjVal())
    print(f"Site-Depot routes for {year2} (from,to): ", depot_routes_year2)
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
    print(f"Optimal refinery transport&utilization cost {year2} = ", model6.getObjVal())
    print(f"Depot-Refinery routes for {year2} (from,to): ", refinery_routes_year2)


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
                          refinery_routes_year2_solution], axis=0)
    solution.to_csv('submission/SCIP_solution.csv')