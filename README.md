
# Shell_hachakton_2023

# Objective
Minimize
1. Cost of transportation $Cost_{transport} $

$$ \large Cost_{transport} = (\sum_{i,j}Dist_{i,j} \cdot Biomass_{i,j}) + (\sum_{j,k} Dist_{j,k} \cdot Pellet_{j,k}) $$

2. Cost of biomass forecast mismatch $Cost_{forecast}$
$$ \large Cost_{forecast} = \sum | Biomass_{forecast,i} - Biomass_{true,i} | $$

3. Cost of underutilization Cost_{underutilization}
$$ \large Cost_{underutilization} = \sum_j (Cap_{depot} - \sum_i Biomass_{i,j}) + \sum_k(Cap_{refinery} - \sum_J Pellet_{j,k}) $$

Overal cost 
$$ \large a \cdot Cost_{transport} + b \cdot Cost_{forecast} + c \cdot Cost_{underutilization} $$
