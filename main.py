import optuna
import pandas as pd

from src.opt import Supply_chain_optimization

bio_hist = pd.read_csv('data/Biomass_history.csv')
dist_matrix = pd.read_csv('data/Distance_matrix.csv', header=None, usecols=[i for i in range(1, (2419))])
sample_submit = pd.read_csv('data/sample_submission.csv')
dist_matrix = dist_matrix.iloc[1:]

dist_arr = dist_matrix.values.copy()
opt = Supply_chain_optimization(dist_arr)

bio_hist = opt.biomass_forecast(bio_hist)
bio_hist['2018'] = bio_hist['2018'].apply(lambda x: x if x > 0 else 0)
bio_hist['2019'] = bio_hist['2019'].apply(lambda x: x if x > 0 else 0)

# mean between 2018 and 2019
last_time_distr = (bio_hist['2018'] + bio_hist['2019']) / 2

n_ref, n_depots = opt.calc_num_objects(last_time_distr)


def optuna_obj(trial):
    values_dep = [trial.suggest_int('depot_idx_{}'.format(i), 0, len(dist_arr) - 1) for i in range((n_depots))]
    values_ref = [trial.suggest_int('ref_idx_{}'.format(i), 0, len(dist_arr) - 1) for i in range((n_ref))]

    opt = Supply_chain_optimization(dist_arr)

    sum_cost_depot = opt.calculate_depot_sum(last_time_distr, values_dep)

    sum_cost_ref = opt.refinery_obj(last_time_distr, values_ref)

    return sum_cost_depot + sum_cost_ref


def get_submission_file(study, path_to_save='submissions/sub1.csv'):
    locations_df = pd.DataFrame.from_dict(study.best_params, orient='index')
    ref_idx = [locations_df.loc[x][0] for x in list(locations_df.index) if 'ref' in x]
    depot_idx = [locations_df.loc[x][0] for x in list(locations_df.index) if 'depot' in x]

    opt = Supply_chain_optimization(dist_arr)

    sum_cost_depot = opt.calculate_depot_sum(last_time_distr, depot_idx)
    sum_cost_ref = opt.refinery_obj(last_time_distr, ref_idx)

    print(f'Sum cost of optimization: {sum_cost_depot + sum_cost_ref}')

    final_df = opt.final_df.copy()

    final_loc = opt.final_df.query("data_type == 'refinery_location' or data_type =='depot_location'")
    final_loc['year'] = 20182019

    # 2018 and 2019 predictions
    dd_2018 = opt.final_df.query("data_type == 'biomass_demand_supply' or data_type =='pallet_demand_supply'")
    dd_2018['year'] = 2018
    dd_2019 = opt.final_df.query("data_type == 'biomass_demand_supply' or data_type =='pallet_demand_supply'")
    dd_2019['year'] = 2019

    biomass_forecast_2018 = bio_hist[['Index', '2018']].rename(columns={'Index': 'source_index', '2018': 'value'})
    biomass_forecast_2018['data_type'] = 'biomass_forecast'
    biomass_forecast_2018['year'] = 2018
    biomass_forecast_2019 = bio_hist[['Index', '2019']].rename(columns={'Index': 'source_index', '2019': 'value'})
    biomass_forecast_2019['data_type'] = 'biomass_forecast'
    biomass_forecast_2019['year'] = 2019

    submission_df = pd.concat([final_loc, biomass_forecast_2018, biomass_forecast_2019, dd_2018, dd_2019])
    submission_df = submission_df[['year', 'data_type', 'source_index', 'destination_index', 'value']]
    submission_df.to_csv(path_to_save)
    return submission_df


if __name__ == 'main':
    study = optuna.create_study()  # Create a new study.
    study.optimize(optuna_obj, n_trials=300)

    get_submission_file(study)
