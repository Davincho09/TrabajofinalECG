import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.metrics import mean_squared_error

# --- Autoencoder simple ---
class SimpleAutoencoder:
    def __init__(self, input_dim, latent_dim, genome):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.W1 = genome[:input_dim*latent_dim].reshape(input_dim, latent_dim)
        self.W2 = genome[input_dim*latent_dim:].reshape(latent_dim, input_dim)

    def forward(self, X):
        h = np.tanh(X @ self.W1)
        return np.tanh(h @ self.W2)

    def mse(self, X):
        return mean_squared_error(X, self.forward(X))

def evolve_autoencoder(X, latent_dim=3, pop_size=30, ngen=20, cxpb=0.6, mutpb=0.2):
    input_dim = X.shape[1]
    n_weights = input_dim * latent_dim + latent_dim * input_dim

    # Si ya existen, evita recrearlos
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)


    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        weights = np.array(ind, dtype=float)
        ae = SimpleAutoencoder(input_dim, latent_dim, weights)
        mse = ae.mse(X)
        complexity = float(np.sum(np.abs(weights))) 
        return float(mse), float(complexity)



    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(10)  # guarda los 10 mejores
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=pop_size, lambda_=pop_size,
        cxpb=cxpb, mutpb=mutpb, ngen=ngen,
        stats=stats, halloffame=hof, verbose=True
    )

    return pop, logbook  
