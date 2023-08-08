import pandas as pd
import numpy as np


class Greedy_algorithm():

    def __init__(self,
            biomass: pd.Series,
            dist: pd.DataFrame,
            number_of_depots = 15,
            number_of_refineries = 3,
            optimize = True,
        ):
        self.biomass = biomass
        self.dist = dist
        self.number_of_depots = number_of_depots
        self.number_of_refineries = number_of_refineries
        self.optimize = optimize

    def cost(self, space, demand, type = 'depot'):
        '''
        If refinery type is chosen, function assumes self.depot_biomass_supply (which is demand for refineries) is calculated
        '''
        if type == 'depot':
            dist = self.dist.iloc[:, space[0:self.number_of_depots]]
            percentage_utilized = space[-1]
        elif type == 'refinery':
            dist = self.dist.iloc[demand.index, space]

        # Distance X Demand
        cost_per_delivery = dist.T * demand

        # Unpivot depot_cost_per_delivery
        melted_df = cost_per_delivery.reset_index().melt(id_vars='index')
        melted_df.columns = ['Destination location', 'Source location', 'Cost']
        melted_df = melted_df[['Source location', 'Destination location', 'Cost']]
        melted_df['Source location'] = melted_df['Source location'].astype('int')
        melted_df['Destination location'] = melted_df['Destination location'].astype('int')
        min_cost_df = melted_df.loc[melted_df.groupby('Source location')['Cost'].idxmin()]
        min_cost_df.reset_index(drop=True, inplace=True)
        min_cost_df = min_cost_df.sort_values(by='Cost')
        if type == 'depot':
            num_rows_to_delete = int((1 - percentage_utilized) * min_cost_df.shape[0])
            min_cost_df = min_cost_df.iloc[:-num_rows_to_delete]

        transport_cost = min_cost_df['Cost'].sum()

        # Calculate biomass/pellet supply
        supply = []
        locations = cost_per_delivery.index.astype('int')
        for location in locations:
            demand_to_supply = min_cost_df.loc[
                min_cost_df['Destination location'] == location, ['Source location']].values.ravel()
            supply.append(demand[demand_to_supply].sum())
        supply = pd.Series(supply, index=locations)

        return transport_cost, supply, min_cost_df

    def depot_constraint(self, space):
        _, depot_biomass_supply, _ = self.cost(space = space,
                                               demand = self.biomass,
                                               type = 'depot')
        if (depot_biomass_supply > 20e3).any():
            return float(1) #constraint violated
        else:
            return float(0) #feasible

    def refinery_constraint(self, space):
        _, refinery_pellet_supply, _ = self.cost(space = space,
                                                 demand = self.depot_biomass_supply,
                                                 type = 'refinery')
        if (refinery_pellet_supply > 10e4).any():
            return float(1) #constraint violated
        else:
            return float(0) #feasible

    def harvest_constraint(self, space):
        _, depot_biomass_supply, _ = self.cost(space=space,
                                               demand=self.biomass,
                                               type='depot')
        if depot_biomass_supply.sum() < 0.8 * self.biomass.sum():
            return float(1) #constraint violated
        else:
            return float(0) #feasible

    def same_location_constraint(self, space):
        """
        Checks if there are no equal integer values in an array.

        Args:
          array: The array to check.

        Returns:
          True if there are no equal integer values in the array, False otherwise.
        """
        seen = set()
        for i in range(len(space)):
            if space[i] in seen:
                return float(1) #constraint violated
            seen.add(space[i])
        return float(0) #feasible

    def integer_constraint(self, space):
        # Map continuous space to integer values
        mapped_space = np.round(space).astype(int)
        return (mapped_space - space)

    def penalty_depot_constraint(self, space):
        _, depot_biomass_supply, _ = self.cost(space=space,
                                               demand=self.biomass,
                                               type='depot')
        if (depot_biomass_supply > 20e3).any():
            depot_overutilization_penalty = 3*(depot_biomass_supply[depot_biomass_supply > 20e3] - 20e3).sum()
        else:
            depot_overutilization_penalty = 0

        return depot_overutilization_penalty

    def penalty_harvest_constraint(self, space):
        _, depot_biomass_supply, _ = self.cost(space=space,
                                               demand=self.biomass,
                                               type='depot')
        if depot_biomass_supply.sum() < 0.8 * self.biomass.sum():
            underharvested_penalty = 3*(0.8 * self.biomass.sum() - depot_biomass_supply.sum())
        else:
            underharvested_penalty = 0

        return underharvested_penalty

    def objective_depot(self, space):

        self.depot_transport_cost, self.depot_biomass_supply, solution = self.cost(space = space,
                                                                                   demand = self.biomass,
                                                                                   type = 'depot')

        # Underutilization cost
        #self.depot_underutilization_cost = (20e3 - self.depot_biomass_supply).sum()
        #self.depot_transport_underutilization_cost = 0.001 * self.depot_transport_cost + self.depot_underutilization_cost

        if self.optimize:
            return 0.001 * self.depot_transport_cost
        else:
            return 0.001 * self.depot_transport_cost, self.depot_biomass_supply, solution


    def objective_refinery(self, space):

        self.refinery_transport_cost, self.refinery_pellet_supply, solution = self.cost(space = space,
                                                                                        demand = self.depot_biomass_supply,
                                                                                        type = 'refinery')

        # Underutilization cost
        self.refinery_underutilization_cost = (10e4 - self.refinery_pellet_supply).sum()
        self.refinery_transport_underutilization_cost = 0.001 * self.refinery_transport_cost + self.refinery_underutilization_cost

        if self.optimize:
            return self.refinery_transport_underutilization_cost
        else:
            return self.refinery_transport_underutilization_cost, self.refinery_pellet_supply, solution
