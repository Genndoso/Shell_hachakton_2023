from skopt.space import Integer, Real


def get_depot_param_space(biomass):
    # at least 80% should be harvested
    number_of_depots = int(0.8 * biomass.sum() / 20000) + 1
    if number_of_depots > 25:
        number_of_depots = 25
    print(f'Optimal number of depots is {number_of_depots}.')
    space = []
    for n in range(number_of_depots):
        space.append(Integer(0, 2417, name=f'Depot {n} location'))
    space.append(Real(0.5, 1, name='Percentage of locations utilized'))

    return space, number_of_depots


def get_refinery_param_space(biomass):
    # all pellets should be delivered
    number_of_refineries = int(biomass.sum() / 100000) + 1

    if number_of_refineries > 5:
        number_of_refineries = 5
    print(f'Optimal number of refineries is {number_of_refineries}.')

    space = []
    for n in range(number_of_refineries):
        space.append(Integer(0, 2417, name=f'Refinery {n} location'))

    return space, number_of_refineries