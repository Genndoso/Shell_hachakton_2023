import pandas as pd
import numpy as np
from sklearn.linear_model import TheilSenRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from pyscipopt import Model, quicksum, multidict
import os
os.chdir('/Users/user/PycharmProjects/Shell_hachakton_2023')
data = pd.read_csv('data/Biomass_History.csv', index_col=0)
dist = pd.read_csv('data/Distance_Matrix.csv', index_col=0)

EPS = 1.e-6
year1 = '2018'; year2 = '2019'
max_number_of_depots = 25
max_number_of_refineries = 5
depot_cap = 20000
refinery_cap = 100000
biomass_percentage_to_collect = 0.85


def preprocessing(data ,dist, biomass_percentage):
    total_biomass = max(data['2018'].sum(), data['2019'].sum())
    average_biomass = data.iloc[:, -2:].mean(axis=1)
    sorted_dist = dist.sum(axis=1).sort_values(ascending=True)
    indices = []
    biomass_to_collect = []
    i = 0
    while sum(biomass_to_collect) <= biomass_percentage * total_biomass:
        biomass_to_collect.append(average_biomass[sorted_dist.index[i]])
        indices.append(sorted_dist.index[i])
        i += 1

    return indices

def cluster_preprocessing(data, num_clusters=5, num_closest_points=10):
    # Selecting the longitude and latitude columns
    print(data)
    X = data[['Longitude', 'Latitude']]

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Find the 5 closest points to each cluster center
    closest_points = []
    closest_points_indices = []

    for cluster_center in cluster_centers:
        # Fit a NearestNeighbors model on the data points belonging to the current cluster
        cluster_indices = (kmeans.labels_ == kmeans.predict([cluster_center])).nonzero()[0]
        cluster_data = X.iloc[cluster_indices]

        nn_model = NearestNeighbors(n_neighbors=num_closest_points)
        nn_model.fit(cluster_data)

        # Find the indices of the closest points in the cluster
        _, indices = nn_model.kneighbors([cluster_center])

        # Append the closest points to the list
        closest_points.extend(cluster_data.iloc[indices[0]].values)

        # Append the closest points' indices to the list
        closest_points_indices.extend(cluster_indices[indices[0]])

    closest_points_indices = np.array(closest_points_indices)

    return cluster_centers, closest_points, closest_points_indices

def predict_biomass(data):
    reg = TheilSenRegressor()
    X = np.arange(0,8).reshape(-1,1)
    pred_list = []
    for i in range(len(data)):
        y = data.iloc[i, 2:].values.ravel()
        reg.fit(X,y)
        preds = reg.predict(np.array([8, 9]).reshape(-1, 1))
        pred_list.append(list(preds))
    pred_df = pd.DataFrame(np.array(pred_list).squeeze(), columns=['2018', '2019'],
                               index=data.index)
    data_predicted = pd.concat([data, pred_df], axis=1)

    return data_predicted
def CFLP(I,J,d,M,c,n):
    model = Model("flp")
    x,y = {},{}

    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)"%j)
        for i in I:
            x[i,j] = model.addVar(vtype="C", name="x(%s,%s)"%(i,j))
    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i) # All demands should be met
    for j in M:
        model.addCons(quicksum(x[i,j] for i in I) <= M[j]*y[j], "Capacity(%s)"%i) # All facility should not violate capacity constraint
    for (i,j) in x:
        model.addCons(x[i,j] <= d[i]*y[j], "Strong(%s,%s)"%(i,j)) # Can't be delivered more than demand
    model.addCons(quicksum(y[j] for j in J) <= n)
    model.setObjective(
        quicksum((M[j]*y[j] - quicksum(x[i,j] for i in I)) for j in J) +
        0.001 * quicksum(c[i,j]*x[i,j] for i in I for j in J),
        "minimize")
    model.data = x,y
    return model

def CFLP_recalculate_routes(I, J, d, M, c, y):
    model = Model("flp")
    x = {}

    for j in J:
        for i in I:
            x[i,j] = model.addVar(vtype="C", name="x(%s,%s)"%(i,j))
    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i) # All demands should be met
    for j in M:
        model.addCons(quicksum(x[i,j] for i in I) <= M[j]*y[j], "Capacity(%s)"%i) # All facility should not violate capacity constraint
    for (i,j) in x:
        model.addCons(x[i,j] <= d[i]*y[j], "Strong(%s,%s)"%(i,j)) # Can't be delivered more than demand
    model.setObjective(
        quicksum((M[j]*y[j] - quicksum(x[i,j] for i in I)) for j in J) +
        0.001 * quicksum(c[i,j]*x[i,j] for i in I for j in J),
        "minimize")
    model.data = x
    return model

def get_routes_solution(model, x, routes, year, data_type):
    solution = np.zeros(shape=(len(routes), 3))
    for i, route in enumerate(routes):
        solution[i, 0] = route[0]
        solution[i, 1] = route[1]
        solution[i, 2] = model.getVal(x[route])
    solution = pd.DataFrame(solution, columns=['source_index', 'destination_index', 'value'])
    solution['year'] = year
    solution['data_type'] = data_type
    solution['source_index'] = solution['source_index'].astype('int')
    solution['destination_index'] = solution['destination_index'].astype('int')
    solution = solution[['year', 'data_type', 'source_index', 'destination_index', 'value']]

    return solution

def get_locations(facilities, year, data_type):
    solution = pd.DataFrame(np.zeros(shape=(len(facilities), 5)),
                            columns=['year', 'data_type', 'source_index', 'destination_index', 'value'])
    solution['year'] = year;
    solution['data_type'] = data_type
    solution['destination_index'] = np.nan;
    solution['value'] = np.nan
    solution['source_index'] = facilities

    return solution

def get_prediction(data, year):
    solution = pd.DataFrame(np.zeros(shape=(len(data), 5)),
                            columns=['year', 'data_type', 'source_index', 'destination_index', 'value'])
    solution['year'] = year;
    solution['data_type'] = 'biomass_forecast'
    solution['source_index'] = data.index
    solution['destination_index'] = np.nan
    solution['value'] = data[year].values

    return solution


if __name__ == '__main__':

    # Prediction
    prediction = predict_biomass(data)
    # Preprocessing
    indices = preprocessing(prediction, dist, biomass_percentage=biomass_percentage_to_collect)
    prediction = prediction.iloc[indices,:]
    dist = dist.iloc[indices, indices]

    # 2018
    # Get site locations and biomass demand
    biomass1 = prediction[year1]
    site_locations, biomass_demand = multidict(dict(biomass1))
    # Possible depot locations
    capacity = pd.Series(depot_cap, index=prediction[year1].index)  # assuming all possible locations
    possible_depot_locations, depot_capacity = multidict(dict(capacity))
    # Cost matrix of transport from site_location to depot_locations
    site_depot_dist = dist.iloc[site_locations, possible_depot_locations].reset_index().melt(id_vars='index')
    site_depot_dist['variable'] = site_depot_dist['variable'].astype('int')
    site_depot_dist = {(row[0], row[1]): row[2] for row in site_depot_dist.values}
    # Optimize depot locations
    model1 = CFLP(I=site_locations,
                  J=possible_depot_locations,
                  d=biomass_demand,
                  M=depot_capacity,
                  c=site_depot_dist,
                  n=max_number_of_depots)
    model1.optimize()
    x1, y1 = model1.data
    depot_routes_year1 = [(i, j) for (i, j) in x1 if model1.getVal(x1[i, j]) > EPS]
    depot_locations = [j for j in y1 if model1.getVal(y1[j]) > EPS]
    print("Optimal value=", model1.getObjVal())
    print("Depots at nodes:", depot_locations)
    print("Site-Depot routes year1 (from,to):", depot_routes_year1)


    # Get pellet demand at each depot
    pellet_demand = {dl: sum([model1.getVal(x1[sl, dl]) for sl in site_locations if model1.getVal(x1[sl, dl]) > EPS]) for
                     dl in depot_locations}

    # Get possible refinery locations
    capacity = pd.Series(refinery_cap, index=prediction[year1].index)  # assuming all possible locations
    possible_refinery_locations, refinery_capacity = multidict(dict(capacity))

    # Cost matrix of transport from depot_locations to refinery_locations
    depot_refinery_dist = dist.iloc[depot_locations, possible_refinery_locations].reset_index().melt(id_vars='index')
    depot_refinery_dist['variable'] = depot_refinery_dist['variable'].astype('int')
    depot_refinery_dist = {(row[0], row[1]): row[2] for row in depot_refinery_dist.values}
    # Optimize refinery locations
    model2 = CFLP(I=depot_locations,
                  J=possible_refinery_locations,
                  d=pellet_demand,
                  M=refinery_capacity,
                  c=depot_refinery_dist,
                  n=max_number_of_refineries)
    model2.optimize()
    x2, y2 = model2.data
    refinery_routes_year1 = [(i, j) for (i, j) in x2 if model2.getVal(x2[i, j]) > EPS]
    refinery_locations = [j for j in y2 if model2.getVal(y2[j]) > EPS]
    print("Optimal value=", model2.getObjVal())
    print("Refineries at nodes:", refinery_locations)
    print("Depot-Refinery routes year1 (from,to):", refinery_routes_year1)


    boolean_depot_locations = {j: model1.getVal(y1[j]) for j in possible_depot_locations}
    boolean_refinery_locations = {j: model2.getVal(y2[j]) for j in possible_refinery_locations}


    # 2019
    biomass2 = prediction[year2]
    site_locations, biomass_demand = multidict(dict(biomass2))
    capacity = pd.Series(depot_cap, index=prediction[year2].index)  # assuming all possible locations
    possible_depot_locations, depot_capacity = multidict(
        dict(capacity))  # len(possible_depot_locations) = len(boolean_depot_locations)
    site_depot_dist = dist.iloc[site_locations, possible_depot_locations].reset_index().melt(id_vars='index')
    site_depot_dist['variable'] = site_depot_dist['variable'].astype('int')
    site_depot_dist = {(row[0], row[1]): row[2] for row in site_depot_dist.values}
    model3 = CFLP_recalculate_routes(I=site_locations,
                                     J=possible_depot_locations,
                                     d=biomass_demand,
                                     M=depot_capacity,
                                     c=site_depot_dist,
                                     y=boolean_depot_locations)
    model3.optimize()
    x3 = model3.data
    depot_routes_year2 = [(i, j) for (i, j) in x3 if model3.getVal(x3[i, j]) > EPS]
    print("Optimal value=", model3.getObjVal())
    print("Site-Depot routes year2 (from,to):", depot_routes_year2)


    # Get pellet demand at each depot
    pellet_demand = {dl: sum([model3.getVal(x3[sl, dl]) for sl in site_locations if model3.getVal(x3[sl, dl]) > EPS])
                     for dl in depot_locations}
    capacity = pd.Series(refinery_cap, index=prediction[year1].index)  # assuming all possible locations
    possible_refinery_locations, refinery_capacity = multidict(dict(capacity))
    depot_refinery_dist = dist.iloc[depot_locations, possible_refinery_locations].reset_index().melt(id_vars='index')
    depot_refinery_dist['variable'] = depot_refinery_dist['variable'].astype('int')
    depot_refinery_dist = {(row[0], row[1]): row[2] for row in depot_refinery_dist.values}
    model4 = CFLP_recalculate_routes(I=site_locations,
                                     J=possible_refinery_locations,
                                     d=pellet_demand,
                                     M=refinery_capacity,
                                     c=depot_refinery_dist,
                                     y=boolean_refinery_locations)
    model4.optimize()
    x4 = model4.data
    refinery_routes_year2 = [(i, j) for (i, j) in x3 if model3.getVal(x3[i, j]) > EPS]
    print("Optimal value=", model3.getObjVal())
    print("Depot-Refinery routes year2 (from,to):", refinery_routes_year2)


    # Compile submission file
    depot_locations_solution = get_locations(facilities=depot_locations,
                                             year='20182019',
                                             data_type='depot_location')
    refinery_locations_solution = get_locations(facilities=refinery_locations,
                                                year='20182019',
                                                data_type='refinery_location')
    biomass_prediction_year1_solution = get_prediction(data=prediction,
                                                       year = year1)
    biomass_prediction_year2_solution = get_prediction(data=prediction,
                                                       year = year2)
    depot_routes_year1_solution = get_routes_solution(model=model1,
                                                      x=x1,
                                                      routes=depot_routes_year1,
                                                      year=year1,
                                                      data_type='biomass_demand_supply')
    refinery_routes_year1_solution = get_routes_solution(model=model2,
                                                         x=x2,
                                                         routes=refinery_routes_year1,
                                                         year=year1,
                                                         data_type='pellet_demand_supply')
    depot_routes_year2_solution = get_routes_solution(model=model3,
                                                      x=x3,
                                                      routes=depot_routes_year2,
                                                      year=year2,
                                                      data_type='biomass_demand_supply')
    refinery_routes_year2_solution = get_routes_solution(model=model4,
                                                         x=x4,
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
                          refinery_routes_year2_solution], axis = 0)
    solution.to_csv('submission/SCIP_solution.csv')
