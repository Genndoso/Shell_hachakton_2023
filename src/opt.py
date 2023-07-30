import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class Supply_chain_optimization:

    def __init__(self, dist_matrix):
        '''

        dist_matrix: np.array
        distance matrix


        '''
        self.DEPOT_LIMIT = 2e4
        self.REF_LIMIT = 1e5
        self.TOL = 200
      #  self.hist = history
        self.dist_arr = dist_matrix.copy()

        self.final_df = pd.DataFrame(columns = ['data_type', 'source_index', 'destination_index', 'value'])
    @staticmethod
    def calc_num_objects(hist: pd.Series):
        '''
        Calculate number of refinaries and depots based on sum values of biomass

        '''
        n_ref = hist.sum() // 10e4
        n_depots = hist.sum() // 20e3
        clip_ref = np.clip(n_ref, 0, 5)
        clip_depots = np.clip(n_depots, 0, 25)
        return int(clip_ref) + 1, int(clip_depots) + 1

    def calculate_depot_sum(self, dep_index, hist):
        distance_arr = self.dist_arr.copy()
        depot_sum = {}
        depot_cost = {}
        for depot in dep_index:
            depot_sum[depot] = hist.iloc[depot]
            depot_cost[depot] = 0
            distance_arr[depot, :] = 1e6

            # depot location
            self.final_df = pd.concat([self.final_df, pd.DataFrame([['depot_location', depot, None, None]],
                                                           columns=self.final_df.columns)], axis = 0)


            while depot_sum[depot] < self.DEPOT_LIMIT - self.TOL:

                arg_indx = np.argmin(distance_arr[:, depot])

                # pallet_demand_supply
                self.final_df = pd.concat([self.final_df, pd.DataFrame([['pallet_demand_supply', depot, arg_indx,
                                                                hist.iloc[arg_indx]]],
                                                                columns=self.final_df.columns)], axis=0)


                if distance_arr[arg_indx, depot] == 1e6:
                    break

                depot_cost[depot] += distance_arr[arg_indx, depot] * hist.iloc[arg_indx]
                distance_arr[arg_indx, :] = 1e6

                depot_sum[depot] += hist.iloc[arg_indx]
            #   biomass_arr[arg_indx] = 0


        sums = [val for val in list(depot_sum.values()) if val > self.DEPOT_LIMIT]

        self.depot_sum = depot_sum

        sum_cost = np.array([value for value in depot_cost.values()]).sum()

        if len(sums) > 0:
            sum_cost += 1e5*len(sums)

        return sum_cost

    def refinery_obj(self, ref_index):

        depots = list(self.depot_sum.keys())
        depots_dict = self.depot_sum.copy()

        distance_arr = self.dist_arr.copy()
        bioref_sum = {}
        bioref_cost = {}
        for bf in ref_index:
            # bioferinery location

            self.final_df = pd.concat([self.final_df, pd.DataFrame([['refinery_location', bf, None, None]],
                                                                   columns=self.final_df.columns)], axis=0)
            done = False

            bioref_sum[bf] = 0
            bioref_cost[bf] = 0
            dep_sum = np.array([val for val in depots_dict.values()]).sum()

            # print(len(depots))
            while not done and len(depots) > 0:
                dist_min = 1e6
                dep_opt = 0
                for i, dep in enumerate(depots):

                    dep_bf_dist = distance_arr[dep, bf]
                    if dep_bf_dist < dist_min:
                        dist_min = dep_bf_dist
                        dep_opt = dep
                        dep_idx = i

                # biomass_demand_supply
                self.final_df = pd.concat([self.final_df, pd.DataFrame([['biomass_demand_supply', dep_opt, bf, depots_dict[dep_opt]]],
                                                                       columns=self.final_df.columns)], axis=0)

                depots.pop(dep_idx)
                bioref_cost[bf] += distance_arr[dep_opt, bf] * depots_dict[dep_opt]

                bioref_sum[bf] += depots_dict[dep_opt]

                if bioref_sum[bf] > self.REF_LIMIT - 1e3:
                    done = True

                depots_dict[dep_opt] = 0


            self.bioref_sum = bioref_sum
            sum_cost = np.array([value for value in bioref_cost.values()]).sum()
        return sum_cost

    def biomass_forecast(self, bio_hist):

        lr = LinearRegression()

        x = np.arange(0,8).reshape(-1,1)

        pred_list = []
        for i in range(0, len(bio_hist)):
            y = bio_hist.iloc[i, 3:].values.reshape(-1, 1)
            lr = LinearRegression().fit(x, y)
            preds = lr.predict(np.array([8, 9]).reshape(-1, 1))
            pred_list.append(list(preds))

        pred_df = pd.DataFrame(np.array(pred_list).squeeze(), columns=['2018', '2019'],
                               index=np.arange(0, len(bio_hist)))

        bio_hist = pd.concat([bio_hist, pred_df], axis=1)
        return bio_hist












