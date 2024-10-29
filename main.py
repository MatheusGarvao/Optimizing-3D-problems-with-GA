import numpy as np
import random as rand
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd


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


# Função para criar populações
# Cria um cromossomo com dois números aleatórios dentro dos limites
# Retorna uma lista de cromossomos
def generate_population(size, bounds):
    cromossomos = []
    for i in range(size):
        cromossomo = [rand.uniform(bounds[0], bounds[1]), rand.uniform(bounds[0], bounds[1])]
        cromossomos.append(cromossomo)
    return cromossomos


# Função de Fitness
def fitness_function(cromossomo, func):
    functions = {0: schwefel, 1: rastrigin, 2: third_function}
    value = functions.get(func, "invalid")(cromossomo)
    return value


# Função de Crossover Aritmético
# Essa função trabalha ao longo das gerações, ela define um peso para o quanto um pai influencia na criação de um novo
# filho, e qual pai vai influênciar mais em qual filho.
def crossover_arithmetic(parent1, parent2, generation, max_generations):
    alpha = rand.uniform(0.5 - 0.5 * (generation / max_generations), 0.5 + 0.5 * (generation / max_generations))
    filho1 = [alpha * parent1[i] + (1 - alpha) * parent2[i] for i in range(len(parent1))]
    filho2 = [(1 - alpha) * parent1[i] + alpha * parent2[i] for i in range(len(parent1))]
    return [filho1, filho2]


# Função de Crossover de Ponto Único
def crossover_single_point(parent1, parent2, generation, max_generations):
    point = rand.randint(1, len(parent1) - 1)
    filho1 = parent1[:point] + parent2[point:]
    filho2 = parent2[:point] + parent1[point:]
    return [filho1, filho2]


# Função de Seleção por Torneio
# Essa função escolhe alguns cromossomos para competirem em um torneio
# neste torneio, ganha quem tiver o melhor fitness dependendo do problema (max ou min).
# O vencedor entra na lista dos selecionados.
# Isso garante que os melhores cromossomos apareçam mais de uma vez na lista dos selecionados
def tournament_selection(population, fitness_scores, tournament_size=3, problem_type='min'):
    selected = []
    for _ in range(len(population)):
        competitors = rand.sample(list(zip(population, fitness_scores)), tournament_size)
        if problem_type == 'min':
            winner = min(competitors, key=lambda x: x[1])
        else:
            winner = max(competitors, key=lambda x: x[1])
        selected.append(winner[0])
    return selected


# Função de Seleção por Roleta
# A roleta é uma roleta que aleatóriamente com um certo peso
# onde individuos com um peso maior tem mais chances de serem selecionados.
def roulette_wheel_selection(population, fitness_scores, problem_type='min'):
    if problem_type == 'min':
        max_fitness = max(fitness_scores)
        inverted_fitness = [max_fitness - f + 1e-6 for f in fitness_scores]
        total_fitness = sum(inverted_fitness)
        if total_fitness == 0:
            selection_probs = [1 / len(population)] * len(population)
        else:
            selection_probs = [f / total_fitness for f in inverted_fitness]
    else:
        min_fitness = min(fitness_scores)
        adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_scores]
        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            selection_probs = [1 / len(population)] * len(population)
        else:
            selection_probs = [f / total_fitness for f in adjusted_fitness]
    selected = []
    for _ in range(len(population)):
        idx = np.random.choice(len(population), p=selection_probs)
        selected.append(population[idx])
    return selected


# Função de Mutação Adaptativa
# É uma função onde a taxa de mutação diminui conforme o tempo, para chegar a um ponto de convergência no final.
def mutate(cromossomo, mutation_rate, bounds, generation, max_generations):
    adapt_mutation_rate = mutation_rate * (1 - generation / max_generations)
    for i in range(len(cromossomo)):
        if rand.random() < adapt_mutation_rate:
            cromossomo[i] = rand.uniform(bounds[0], bounds[1])
    return cromossomo


# Função de Evolução com Elitismo de 25%
# Essa função serve para evoluir a população e selecionar os 25% melhores,
# além de aplicar o crossover e mutações
def evolve_population(population, fitness_func, mutation_rate, bounds, selection_func, crossover_func, generation,
                     max_generations, problem_type):
    # Calcula o fitness de cada cromossomo na população
    fitnesses = [fitness_function(cromossomo, fitness_func) for cromossomo in population]

    # Combina a população com seus fitnesses
    population_fitness = list(zip(population, fitnesses))

    # Determina o tamanho da elite (25% da população)
    elite_size = int(0.25 * len(population))

    # Ordena a população com base no fitness
    if problem_type == 'min':
        # Para minimização, menores fitness são melhores
        sorted_population = sorted(population_fitness, key=lambda x: x[1])
    else:
        # Para maximização, maiores fitness são melhores
        sorted_population = sorted(population_fitness, key=lambda x: x[1], reverse=True)

    # Separa os elites (os melhores cromossomos)
    elites = [cromossomo for cromossomo, fitness in sorted_population[:elite_size]]

    # Seleciona a população para a próxima geração usando o método de seleção escolhido
    selected_population = selection_func(population, fitnesses, problem_type=problem_type)

    # Aplica o crossover para criar novos cromossomos
    population_after_crossover = []
    for i in range(0, len(selected_population), 2):
        parent1 = selected_population[i]
        if i + 1 < len(selected_population):
            parent2 = selected_population[i + 1]
        else:
            parent2 = selected_population[0]  # Se for ímpar, emparelha com o primeiro
        filhos = crossover_func(parent1, parent2, generation, max_generations)
        population_after_crossover.extend(filhos)

    # Aplica a mutação nos novos cromossomos
    population_after_mutation = []
    for i in range(len(population_after_crossover)):
        cromossomo_mutado = mutate(population_after_crossover[i], mutation_rate, bounds, generation, max_generations)
        population_after_mutation.append(cromossomo_mutado)

    # Remove os elites da população mutada para fazer espaço
    # Isso garante que o tamanho total da população permaneça constante
    population_after_mutation = population_after_mutation[:len(population) - elite_size]

    # Combina os elites preservados com a nova população após crossover e mutação
    new_population = elites + population_after_mutation

    # Retorna a nova população que inclui os elites preservados
    return new_population


# Função para plotar a superfície 3D da função a ser otimizada
def plot_3d_surface(func, bounds, func_name, optimal_points=None):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([x_val, y_val]) for x_val in x] for y_val in y])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax.set_title(f'Surface Plot of {func_name}')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis (fitness)')

    if optimal_points:
        for point in optimal_points:
            x_opt, y_opt = point['cromossomo']
            z_opt = func(point['cromossomo'])
            ax.scatter(x_opt, y_opt, z_opt, color='r', s=50)
            ax.text(x_opt, y_opt, z_opt, f"{point['label']}", color='black')

    plt.show()


# Função para exibir a evolução do fitness ao longo das gerações
def plot_fitness_evolution(fitness_history, func_name, sel_name, cross_name):
    generations = range(len(fitness_history))
    max_fitness = [np.max(fitness) for fitness in fitness_history]
    min_fitness = [np.min(fitness) for fitness in fitness_history]
    avg_fitness = [np.mean(fitness) for fitness in fitness_history]
    std_fitness = [np.std(fitness) for fitness in fitness_history]

    plt.figure()
    plt.plot(generations, max_fitness, label="Max Fitness", color='r')
    plt.plot(generations, min_fitness, label="Min Fitness", color='b')
    plt.plot(generations, avg_fitness, label="Avg Fitness", color='g')
    plt.fill_between(generations, np.array(avg_fitness) - np.array(std_fitness),
                     np.array(avg_fitness) + np.array(std_fitness), color='gray', alpha=0.2, label='Std Dev')

    plt.xlabel('Gerações')
    plt.ylabel('Fitness')
    plt.title(f'Evolução do Fitness\n{func_name} com {sel_name} e {cross_name}')
    plt.legend()
    plt.show()


# Dicionário de funções
functions = {0: schwefel, 1: rastrigin, 2: third_function}


# Função para rodar o AG e plotar os gráficos
def run_and_plot(func, bounds, func_name):
    # Parâmetros do Algoritmo Genético
    g_pop_size = 1000          # Número de cromossomos na população
    g_num_generations = 1000  # Número de vezes que o AG vai tentar melhorar
    g_mutation_rate = 0.45    # Chance de mudar um gene

    problem_type = 'min'
    optimal_fitness = 0
    tolerance = 0.1
    if func == 2:
        # Se for a terceira função, queremos maximizar
        problem_type = 'max'
        optimal_fitness = 0.4289
        tolerance = 0.0001

    # Métodos de seleção disponíveis
    selection_methods = {
        'Tournament Selection': tournament_selection,  # Seleção por torneio
        'Roulette Wheel Selection': roulette_wheel_selection  # Seleção por roleta
    }

    # Métodos de crossover disponíveis
    crossover_methods = {
        'Arithmetic Crossover': crossover_arithmetic,  # Crossover aritmético
        'Single-point Crossover': crossover_single_point  # Crossover de ponto único
    }

    results = {}  # Guarda os resultados de cada combinação de métodos
    all_final_fitnesses = {}  # Guarda todos os fitness finais para análise paramétrica

    for sel_name, sel_func in selection_methods.items():
        for cross_name, cross_func in crossover_methods.items():
            print(f"\nExecutando AG com {sel_name} e {cross_name} para {func_name}")
            convergence_generations = []
            final_fitnesses = []  # Lista para armazenar os fitness finais
            best_overall_fitness = None
            best_overall_cromossomo = None
            best_fitness_history = None
            for run in range(10):
                print(f"\nExecução {run + 1} de 10")

                g_population = generate_population(g_pop_size, bounds)
                fitness_history = []
                best_fitness_ever = None
                converged = False
                for generation in range(g_num_generations):
                    g_population = evolve_population(g_population, func, g_mutation_rate, bounds, sel_func, cross_func, generation, g_num_generations, problem_type)
                    print(len(g_population))
                    g_fitnesses = [fitness_function(cromossomo, func) for cromossomo in g_population]
                    fitness_history.append(g_fitnesses)
                    if problem_type == 'min':
                        best_fitness = min(g_fitnesses)
                        best_cromossomo = g_population[g_fitnesses.index(best_fitness)]
                    else:
                        best_fitness = max(g_fitnesses)
                        best_cromossomo = g_population[g_fitnesses.index(best_fitness)]
                    print(f"Geração {generation + 1}, Melhor Fitness: {best_fitness}, Melhor Cromossomo: {best_cromossomo}")

                    if (best_fitness_ever is None) or (problem_type == 'min' and best_fitness < best_fitness_ever) or (problem_type == 'max' and best_fitness > best_fitness_ever):
                        best_fitness_ever = best_fitness

                    if abs(best_fitness - optimal_fitness) <= tolerance:
                        converged = True
                        print(f"Atingiu o valor ótimo na geração {generation + 1}")
                        convergence_generations.append(generation + 1)
                        break

                if not converged:
                    convergence_generations.append(g_num_generations)

                # Armazenar o fitness final da execução
                final_fitnesses.append(best_fitness_ever)

                if (best_overall_fitness is None) or (problem_type == 'min' and best_fitness_ever < best_overall_fitness) or (problem_type == 'max' and best_fitness_ever > best_overall_fitness):
                    best_overall_fitness = best_fitness_ever
                    best_overall_cromossomo = best_cromossomo
                    best_fitness_history = fitness_history

            # Calcula a média das gerações para convergência
            media_generations = np.mean(convergence_generations)
            print(f"\nMédia de gerações para convergência: {media_generations:.2f}")

            # Armazena os fitness finais para análise paramétrica
            all_final_fitnesses[f"{sel_name} + {cross_name}"] = final_fitnesses

            results[(sel_name, cross_name)] = {
                'convergence_generations': convergence_generations,
                'best_overall_fitness': best_overall_fitness,
                'best_overall_cromossomo': best_overall_cromossomo,
                'fitness_history': best_fitness_history
            }

            plot_fitness_evolution(best_fitness_history, func_name, sel_name, cross_name)

    # Depois de todas as combinações, plota um gráfico mostrando quantas gerações cada combinação demorou para
    # encontrar a solução
    boxplot_data = []
    labels = []
    for (sel_name, cross_name), data in results.items():
        boxplot_data.append(data['convergence_generations'])
        labels.append(f'{sel_name}\n{cross_name}')

    plt.figure(figsize=(8, 6))
    plt.boxplot(boxplot_data, tick_labels=labels)
    plt.title(f'Gerações para Convergência - {func_name}')
    plt.ylabel('Gerações para Convergência')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=list(all_final_fitnesses.values()))
    plt.xticks(ticks=range(len(all_final_fitnesses)), labels=list(all_final_fitnesses.keys()), rotation=45)
    plt.title(f'Fitness Finais por Algoritmo - {func_name}')
    plt.ylabel('Fitness Final')
    plt.tight_layout()
    plt.show()

    data = [fitness for fitness in all_final_fitnesses.values()]
    labels_anova = list(all_final_fitnesses.keys())

    F, p = stats.f_oneway(*data)
    print(f'\nAnálise ANOVA para {func_name}:')
    print(f'  Estatística F: {F:.4f}')
    print(f'  Valor-p: {p:.4f}')
    if p < 0.05:
        print('  Resultado: Há diferenças significativas entre os algoritmos.')
    else:
        print('  Resultado: Não há diferenças significativas entre os algoritmos.')

    plt.figure(figsize=(10, 6))
    for label, fitnesses in all_final_fitnesses.items():
        sns.kdeplot(fitnesses, label=label, fill=True)
    plt.title(f'Distribuição dos Fitness Finais - {func_name}')
    plt.xlabel('Fitness Final')
    plt.ylabel('Densidade')
    plt.legend()
    plt.tight_layout()
    plt.show()

    optimal_points = []
    for (sel_name, cross_name), data in results.items():
        optimal_points.append({
            'cromossomo': data['best_overall_cromossomo'],
            'fitness': data['best_overall_fitness'],
            'label': f"{sel_name} + {cross_name}"
        })

    plot_3d_surface(functions[func], bounds, func_name, optimal_points=optimal_points)

    for (sel_name, cross_name), data in results.items():
        print(f"\nMelhor fitness para {func_name} com {sel_name} e {cross_name}: {data['best_overall_fitness']}")
        print(f"Melhor cromossomo: {data['best_overall_cromossomo']}")


run_and_plot(0, [-500, 500], "Schwefel")
run_and_plot(1, [-5, 5], "Rastrigin")
run_and_plot(2, [-2, 2], "Third Function")
