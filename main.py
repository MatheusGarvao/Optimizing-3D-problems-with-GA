import numpy as np
import random as rand
import matplotlib.pyplot as plt


# Função Schwefel
def schwefel(x):
    valor = 0
    for i in range(len(x)):
        valor += x[i] * np.sin(np.sqrt(abs(x[i])))
    return 418.9829 * len(x) - valor


# Função Rastrigin
def rastrigin(x):
    return 20 + np.pow(x[0], 2) + np.pow(x[1], 2) - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))


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
    value = functions.get(func, "invalid")(cromossomo)
    if func == 2:
        return -value  # Maximização da terceira função
    return value


# Função de Seleção com Elitismo
def select_best(population, fitness_scores, num_best):
    population_with_fitness_scores = list(zip(population, fitness_scores))
    population_with_fitness_scores = sorted(population_with_fitness_scores, key=lambda x: x[1])
    num_best = min(num_best, len(population))
    elite_population = [cromossomo for cromossomo, fitness in population_with_fitness_scores[:num_best]]
    return elite_population


# Função de Crossover (agora aritmético)
def crossover(parent1, parent2, generation, max_generations):
    alpha = rand.uniform(0.5 - 0.5 * (generation / max_generations), 0.5 + 0.5 * (generation / max_generations))
    filho1 = [alpha * parent1[i] + (1 - alpha) * parent2[i] for i in range(len(parent1))]
    filho2 = [(1 - alpha) * parent1[i] + alpha * parent2[i] for i in range(len(parent1))]
    return [filho1, filho2]


# Função de Mutação Adaptativa
def mutate(cromossomo, mutation_rate, bounds, generation, max_generations):
    adapt_mutation_rate = mutation_rate * (1 - generation / max_generations)  # Diminui a mutação ao longo do tempo
    for i in range(len(cromossomo)):
        if rand.random() < adapt_mutation_rate:
            cromossomo[i] = rand.uniform(bounds[0], bounds[1])
    return cromossomo


# Função de Evolução
def evolve_population(population, fitness_func, mutation_rate, bounds, size, generation, max_generations):
    fitnesses = [fitness_function(cromossomo, fitness_func) for cromossomo in population]

    population = select_best(population, fitnesses, size)

    population_after_crossover = []
    for i in range(len(population) // 2):
        parent1 = population[i]
        parent2 = population[len(population) - i - 1]
        filhos = crossover(parent1, parent2, generation, max_generations)
        population_after_crossover.extend(filhos)

    population_after_mutation = []
    for i in range(len(population_after_crossover)):
        cromossomo_mutado = mutate(population_after_crossover[i], mutation_rate, bounds, generation, max_generations)
        population_after_mutation.append(cromossomo_mutado)

    return population_after_mutation


# Função para plotar a superfície 3D da função a ser otimizada
def plot_3d_surface(func, bounds, func_name):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([x_val, y_val]) for x_val in x] for y_val in y])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title(f'Surface Plot of {func_name}')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis (fitness)')
    plt.show()


# Função para exibir a evolução do fitness ao longo das gerações
def plot_fitness_evolution(fitness_history):
    generations = range(len(fitness_history))
    max_fitness = [np.max(fitness) for fitness in fitness_history]
    min_fitness = [np.min(fitness) for fitness in fitness_history]
    avg_fitness = [np.mean(fitness) for fitness in fitness_history]

    plt.plot(generations, max_fitness, label="Max Fitness", color='r')
    plt.plot(generations, min_fitness, label="Min Fitness", color='b')
    plt.plot(generations, avg_fitness, label="Avg Fitness", color='g')

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution Over Generations')
    plt.legend()
    plt.show()


# Função para rodar o AG e plotar os gráficos para diferentes funções e limites
def run_and_plot(func, bounds, func_name):
    g_pop_size = 10000
    g_num_generations = 100000
    g_mutation_rate = 0.2
    num_best_to_select = 2500
    no_improvement_generations = 0
    best_fitness_ever = float('inf')

    g_population = generate_population(g_pop_size, bounds)
    fitness_history = []

    for generation in range(g_num_generations):
        g_population = evolve_population(g_population, func, g_mutation_rate, bounds, num_best_to_select, generation, g_num_generations)

        # Coleta dos valores de fitness para cada geração
        g_fitnesses = [fitness_function(cromossomo, func) for cromossomo in g_population]
        fitness_history.append(g_fitnesses)

        best_fitness = min(g_fitnesses)
        best_cromossomo = g_population[g_fitnesses.index(best_fitness)]

        g_population.extend(generate_population(g_pop_size-num_best_to_select, bounds))
        print(f"Geração {generation + 1}, Melhor Fitness: {best_fitness}, Melhor Cromossomo: {best_cromossomo}")

        # Condição de parada se o melhor fitness é próximo de 0 (para Schwefel)
        if func == 0 and np.isclose(best_fitness, 0, atol=0.1):
            print("Atingiu o valor ótimo!")
            break

        # Verificar estagnação
        if best_fitness < best_fitness_ever:
            best_fitness_ever = best_fitness
            no_improvement_generations = 0  # Resetar se houver melhora
        else:
            no_improvement_generations += 1

    # Plotar a evolução do fitness
    plot_fitness_evolution(fitness_history)

    # Plotar a superfície 3D da função
    plot_3d_surface(functions.get(func), bounds, func_name)


# Dicionário de funções
functions = {0: schwefel, 1: rastrigin, 2: third_function}

run_and_plot(0, [-500, 500], "Schwefel")
#run_and_plot(1, [-5, 5], "Rastrigin")
#run_and_plot(2, [-2, 2], "Third Function")
