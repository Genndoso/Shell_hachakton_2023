import pandas as pd


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


    def objective_depot(self, space):

        depot_dist = self.dist.iloc[:, space[0:self.number_of_depots]]
        percentage_utilized = space[-1]

        # Distance X Demand
        depot_cost_per_delivery = depot_dist.T * self.biomass

        # Unpivot depot_cost_per_delivery
        melted_df = depot_cost_per_delivery.reset_index().melt(id_vars='index')
        melted_df.columns = ['Destination location', 'Source location', 'Cost']
        melted_df = melted_df[['Source location', 'Destination location', 'Cost']]
        melted_df['Source location'] = melted_df['Source location'].astype('int')
        melted_df['Destination location'] = melted_df['Destination location'].astype('int')
        min_cost_df = melted_df.loc[melted_df.groupby('Source location')['Cost'].idxmin()]
        min_cost_df.reset_index(drop=True, inplace=True)
        sorted_df = min_cost_df.sort_values(by='Cost')
        num_rows_to_delete = int((1 - percentage_utilized) * sorted_df.shape[0])
        trimmed_df = sorted_df.iloc[:-num_rows_to_delete]

        self.overall_depot_delivery_cost = trimmed_df['Cost'].sum()

        # Calculate biomass supply delivered to each depot
        self.depot_biomass_supply = []
        depots = depot_cost_per_delivery.index.astype('int')
        for depot in depots:
            HS_to_depot = trimmed_df.loc[
                trimmed_df['Destination location'] == depot, ['Source location']].values.ravel()
            self.depot_biomass_supply.append(self.biomass[HS_to_depot].sum())
        self.depot_biomass_supply = pd.Series(self.depot_biomass_supply, index=depots)

        if self.optimize:
            # Each depot is max 20000 capacity
            if (self.depot_biomass_supply > 20e3).any():
                return 10e3*sorted_df['Cost'].sum()

            # More than 80% of biomass should be harvested
            elif 0.8 * self.biomass.sum() > self.depot_biomass_supply.sum():
                return 10e3*sorted_df['Cost'].sum()

            else:
                return self.overall_depot_delivery_cost
        else:
            return self.overall_depot_delivery_cost, self.depot_biomass_supply, trimmed_df


    def objective_refinery(self, space):

        refinery_dist = self.dist.iloc[self.depot_biomass_supply.index, space]

        # Distance X Demand
        refinery_cost_per_delivery = refinery_dist.T * self.depot_biomass_supply

        # Unpivot refinery_cost_per_delivery
        melted_df = refinery_cost_per_delivery.reset_index().melt(id_vars='index')
        melted_df.columns = ['Destination location', 'Source location', 'Cost']
        melted_df = melted_df[['Source location', 'Destination location', 'Cost']]
        melted_df['Source location'] = melted_df['Source location'].astype('int')
        melted_df['Destination location'] = melted_df['Destination location'].astype('int')
        min_cost_df = melted_df.loc[melted_df.groupby('Source location')['Cost'].idxmin()]
        min_cost_df.reset_index(drop=True, inplace=True)

        self.overall_refinery_delivery_cost = min_cost_df['Cost'].sum()

        # Calculate biomass supply delivered to each refinery
        self.refinery_biomass_supply = []
        refineries = refinery_cost_per_delivery.index.astype('int')
        for refinery in refineries:
            depot_to_refinery = min_cost_df.loc[
                min_cost_df['Destination location'] == refinery, ['Source location']].values.ravel()
            self.refinery_biomass_supply.append(self.depot_biomass_supply[depot_to_refinery].sum())
        self.refinery_biomass_supply = pd.Series(self.refinery_biomass_supply, index=refineries)

        if self.optimize:
            # Each refinery is max 100000 capacity
            if (self.refinery_biomass_supply > 10e4).any():
                return 10e3 * self.overall_refinery_delivery_cost

            else:
                return self.overall_refinery_delivery_cost
        else:
            return self.overall_refinery_delivery_cost, self.refinery_biomass_supply, min_cost_df