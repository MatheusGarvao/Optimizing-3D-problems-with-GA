import numpy as np
import random as rand

# Função Schwefel
def schwefel(x):
    valor = 0
    for i in range(len(x)):
        valor += x[i] * np.sin(np.sqrt(abs(x[i])))
    return 418.9829 * len(x) - valor


# Função Rastrigin
def rastrigin(x):
    return 20 + np.pow(x[0], 2) + np.pow(x[1], 2) - 10 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))


# Terceira Função (maximização)
def third_function(x):
    return x[0] * np.exp(-(np.pow(x[0], 2) + np.pow(x[1], 2)))


# Função para gerar a população inicial
def generate_population(size, bounds):
    cromossomos = []
    for i in range(size):
        cromossomo = [rand.uniform(bounds[0], bounds[1]), rand.uniform(bounds[0], bounds[1])]
        cromossomos.append(cromossomo)
    return cromossomos


# Função para avaliar a fitness de um cromossomo
def fitness_function(cromossomo, func):
    functions = {0: schwefel, 1: rastrigin, 2: third_function}
    return functions.get(func, "invalid")(cromossomo)


# Função de Seleção
def select_best(population, fitness_scores, num_best):
    population_with_fitness_scores = list(zip(population, fitness_scores))
    population_with_fitness_scores = sorted(population_with_fitness_scores, key=lambda x: x[1])
    num_best = min(num_best, len(population))
    return [cromossomo for cromossomo, fitness in population_with_fitness_scores[:num_best]]


# Função de Crossover
def crossover(parent1, parent2):
    pass


# Função de Mutação
def mutate(cromossomo, mutation_rate, bounds):
    pass


# Função de Evolução
def evolve_population(population, fitness_func, mutation_rate, bounds):
    pass


# Função para plotar a superfície 3D da função a ser otimizada
def plot_3d_surface(func, bounds):
    pass


# Função para exibir a evolução do fitness ao longo das gerações
def plot_fitness_evolution(fitness_history):
    pass
