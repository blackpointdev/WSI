import math
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


def correct_solution(w, v, W, sol):
    num = w.shape[0]
    while w[sol].sum() > W:
        indx = np.random.randint(num)
        while sol[indx%num] == False:
            indx = indx + 1
        sol[indx%num] = False

def correct_solution2(w, v, W, sol):
    ratio = v/w
    num = w.shape[0]
    while w[sol].sum() > W:
        while True:
            indx = np.argmin(ratio)
            if sol[indx] == False:
                ratio[indx] = 10000000
            else:
                sol[indx] = False
                break

def generate_problem(wmin, wmax, vmin, vmax, items_num):
    w = np.random.randint(wmin, wmax, size=items_num)  #weight
    v = np.random.randint(vmin, vmax, size=items_num)  #values
    return w, v

def gen_population(w, v, W, pop_size, corr_method = 1):
    '''
    Method for generating population.
    Input:
        w - list of weights of items in knapsack
        v - list of values of items in knapsack
        W - capacity of knapsack
        pop_size - size of generated population should be
        corr_method - index of correction method (1 - standard method, 2 - improved)
    '''
    population = np.random.randint(0, 2, size=(pop_size, len(w)), dtype=bool)
    for individual in population:
        if corr_method == 1: # Basic, random method
            individual = correct_solution(w, v, W, individual)
        elif corr_method == 2: # Custom, improved method
            individual = correct_solution2(w, v, W, individual)

    return population


def to_decimal(ind, B):
    dec = sum([ind[-n - 1] * (2 ** n) for n in range(B)])
    return dec


def decode_individual(individual, N, B, a, dx):
    decode_individual = np.array(
        [(a + to_decimal(individual[n * B:n * B + B], B)) * dx for n in range(len(individual) // B)])
    return decode_individual


def evaluate_population(population, v):
    '''
    Method for evaluating population.
    :param population: population generated before
    :param v: list of values of items in knapsack
    '''
    evaluated_pop = np.ndarray(shape = (1, len(population)))
    for i in evaluated_pop:
        i = np.sum(v[population[i]]) # Not sure if it works
    return evaluated_pop


def get_best(pop, evaluated_pop):
    best_value = np.amax(evaluated_pop)
    best_individual = np.array(pop[np.argmax(evaluated_pop)])
    return best_individual, best_value


def roulette(pop, evaluated_pop):
    '''
    Method for selection from population
    :param pop: Population
    :param evaluated_pop:
    :return:
    '''
    new_pop = np.copy(pop)
    for i in range(len(evaluated_pop)):
        rand = np.random.random()
        for j in range(len(evaluated_pop)):
            if evaluated_pop[j] > rand:
                new_pop[i] = pop[j]
                break
    return new_pop


def cross(pop, pk, w, v, W):
    '''
    Method for crossing given population.
    :param pop: Given population
    :param pm: Probability of crossing
    :param w: List of weights of items in knapsack
    :param v: List of values of items in knapsack
    :param W: Capacity of knapsack
    :param corr_method: index of correction method (1 - standard method, 2 - improved)
    :return: Crossed population.
    '''
    # TODO Not finished!!!
    new_pop = np.ndarray(shape=(len(pop), len(pop[0])), dtype="int")
    for i in range(0, len(pop) - 1, 2):
        if np.random.random() < pk:
            cross_p = len(pop[0] / 2)
            for j in range(cross_p):
                new_pop[i][j] = pop[i][j]
                new_pop[i + 1][j] = pop[i + 1][j]
            for k in range(cross_p, len(pop[0])):
                new_pop[i][j] = pop[i + 1][j]
                new_pop[i + 1][j] = pop[i][j]
        else:
            new_pop[i] = pop[i]
            new_pop[i + 1] = pop[i + 1]
        if len(pop) % 2 == 1:
            new_pop[len(pop) - 1] = pop[len(pop) - 1]
    return new_pop


def mutate(pop, pm, w, v, W, corr_method = 1):
    '''
    Method for mutating given population.
    :param pop: Given population
    :param pm: Probability of mutation
    :param w: List of weights of items in knapsack
    :param v: List of values of items in knapsack
    :param W: Capacity of knapsack
    :param corr_method: index of correction method (1 - standard method, 2 - improved)
    :return: Mutated population
    '''
    for i in range(len(pop)):
        for j in range(len(pop[i])):
            if np.random.rand() <= pm:
                pop[i][j] = not pop[i][j]

        if corr_method == 1:
            pop[i] = correct_solution(w, v, W, pop[i])
        elif corr_method:
            pop[i] = correct_solution2(w, v, W, pop[i])
    return pop

def genetic_evolution(w, v, W, pop_size, pk, pm, generations, plot):
    N = 2
    pop = gen_population(w, v, W, pop_size)
    best_generation = 1
    list_best = []
    list_best_generation = []
    list_mean = []
    evaluated_pop = evaluate_population(fun, pop, 2, B, -10, dx)
    best_sol = get_best(pop, evaluated_pop)

    first_pop_sol = np.apply_along_axis(decode_individual, 1, pop, N, B, -10, dx)
    first_pop_eval = np.copy(evaluated_pop)

    for i in range(2, generations + 1):
        pop = roulette(pop, evaluated_pop)
        pop = cross(pop, pk)
        pop = mutate(pop, pm)

        evaluated_pop = evaluate_population(fun, pop, 2, B, -10, dx)
        best_sol_temp = get_best(pop, evaluated_pop)

        list_best_generation.append(-best_sol_temp[1])
        list_mean.append(np.mean(-evaluated_pop))

        if best_sol_temp[1] > best_sol[1]:
            best_sol = best_sol_temp
            best_generation = i

        list_best.append(-best_sol[1])

        if i == generations // 2:
            mid_pop_sol = np.apply_along_axis(decode_individual, 1, pop, N, B, -10, dx)
            mid_pop_eval = np.copy(evaluated_pop)

        if i == generations:
            last_pop_sol = np.apply_along_axis(decode_individual, 1, pop, N, B, -10, dx)
            last_pop_eval = np.copy(evaluated_pop)

    best_x, best_y = decode_individual(best_sol[0], 2, B, -10, dx)
    best_z = best_sol[1]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(-20, 20, 0.25)
        y = np.arange(-20, 20, 0.25)
        x, y = np.meshgrid(x, y)
        z = np.array([-obj_func([x_z, y_z]) for x_z, y_z in zip(np.ravel(x), np.ravel(y))])
        z = z.reshape(x.shape)
        surf = ax.plot_surface(x, y, z, cmap='Blues', linewidth=0)

        ax.scatter(-first_pop_sol.T[0], -first_pop_sol.T[1], -first_pop_eval, label='First Gen', c="b")
        ax.scatter(-mid_pop_sol.T[0], -mid_pop_sol.T[1], -mid_pop_eval, label='Mid Gen', c="g")
        ax.scatter(-last_pop_sol.T[0], -last_pop_sol.T[1], -last_pop_eval, label='Last Gen', c="r")

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Levy function (x, y)')
        ax.legend()

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    return best_sol, best_generation, list_best, list_best_generation, list_mean