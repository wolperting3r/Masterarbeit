from data import generate_data
# from models.ML import learning

# generate_data(N_values=1, st_sz=[5, 3])
# generate_data(N_values=1e6, st_sz=[7, 3], equal_kappa=False)
# generate_data(N_values=1e6, st_sz=[3, 3], equal_kappa=False)

generate_data(N_values=1e6, st_sz=[5, 5], equal_kappa=False)
# generate_data(N_values=1e6, st_sz=[7, 3], equal_kappa=True)
# generate_data(N_values=1e6, st_sz=[3, 3], equal_kappa=True)
# generate_data(N_values=1e6, st_sz=[5, 5], equal_kappa=True)
# learning()
