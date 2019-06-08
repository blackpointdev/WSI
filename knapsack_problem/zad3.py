import math
import numpy as np
import time
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

def get_random_solution(w, v, W):
    num = w.shape[0]
    sol = np.random.randint(0,2, size=num, dtype=np.bool)  # 1 / True oznacza, ze przedmiot jest wybrany
    _V = np.sum(v[sol])
    _W = np.sum(w[sol])
    if _W > W:
        correct_solution(w,v,W,sol)
        _V = np.sum(v[sol])
        _W = np.sum(w[sol])
    return sol, _W, _V

def search_random(w,v,W,iters):
    best_sol, best_W, best_V = get_random_solution(w,v,W)
    v_all = [best_V]
    v_best = [best_V]
    for i in range(iters):
        sol, _W, _V = get_random_solution(w,v,W)
        if best_V < _V:
            best_sol, best_W, best_V = sol, _W, _V
        v_all.append(_V)
        v_best.append(best_V)
#     plt.figure()
#     plt.plot(v_all)
#     plt.plot(v_best)
#     plt.show()
    return best_sol, best_W, best_V, v_all, v_best

def search_greedy_improvement(w, v, W, iters):
    best_sol, best_W, best_V = get_random_solution(w,v,W)
    v_all = [best_V]
    v_best = [best_V]
    num = w.shape[0]
    for i in range(iters):
        sol = best_sol.copy()
        #set random 0 bit to 1
        indx = np.random.randint(num)
        while sol[indx%num] == True:
            indx = indx + 1
        sol[indx%num] = True
        #correct if needed
        if w[sol].sum() > W:
            correct_solution(w,v,W,sol)
        _V = v[sol].sum()
        _W = w[sol].sum()
        if best_V < _V:
            best_sol, best_W, best_V = sol.copy(), _W, _V
        v_all.append(_V)
        v_best.append(best_V)
#     plt.figure()
#     plt.plot(v_all)
#     plt.plot(v_best)
#     plt.show()
    return best_sol, best_W, best_V, v_all, v_best

#Pakuje najpierw najbardziej wartościowe przedmioty
def get_value_first(w, v, W):
    ii = np.argsort(-v)
    num = w.shape[0]
    sol = np.repeat(False, num)
    _W = 0
    for i in range(num):
        if _W + w[ii[i]] <= W:
            sol[ii[i]] = True
            _W = _W + w[ii[i]]
    _V = v[sol].sum()
    return sol, _W, _V

def get_ratio_first(w, v, W):
    ii = np.argsort(-v/w) #stosunek wartosci do wagi
    num = w.shape[0]
    sol = np.repeat(False, num)
    _W = 0
    for i in range(num):
        if _W + w[ii[i]] <= W:
            sol[ii[i]] = True
            _W = _W + w[ii[i]]
    _V = v[sol].sum()
    return sol, _W, _V

def generate_problem(wmin, wmax, vmin, vmax, items_num):
    w = np.random.randint(wmin, wmax, size=items_num)  #weight
    v = np.random.randint(vmin, vmax, size=items_num)  #values
    return w, v

def gen_pop(w, v, W, pop_size, corr_method = 1):
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

def evaluate(population, v):
    '''
    Method for evaluating population.
    :param population: population generated before
    :param v: list of values of items in knapsack
    '''
    evaluated_pop = np.ndarray(shape = (len(population),))
    for i in range(len(evaluated_pop)):
        evaluated_pop[i] = np.sum(v[population[i]])
    return evaluated_pop

def select(pop, evaluated_pop):
    '''
    Method for selection from population
    :param pop: Population
    :param evaluated_pop:
    :return:
    '''
    new_pop = np.copy(pop)
    sum = evaluated_pop.sum()
    probability = np.ndarray(shape=len(evaluated_pop), dtype="float")
    probability[0] = evaluated_pop[0]/sum
    for i in range(len(evaluated_pop)-1):
        probability[i+1] = evaluated_pop[i+1] / sum + probability[i]
    for i in range(len(evaluated_pop)):
        rand = np.random.random()
        for j in range(len(evaluated_pop)):
            if probability[j] > rand:
                new_pop[i] = pop[j]
                break
    return new_pop


def xover(pop, p, w, v, W, corr_method = 1):
    '''
    Method for crossing given population.
    :param pop: Given population
    :param p: Probability of crossing
    :param w: List of weights of items in knapsack
    :param v: List of values of items in knapsack
    :param W: Capacity of knapsack
    :param corr_method: index of correction method (1 - standard method, 2 - improved)
    :return: Crossed population.
    '''
    new_pop = np.copy(pop)
    pop_len = len(pop)
    for i in range(0, pop_len - 1, 2):
        point = int(np.random.rand() * (len(pop[i]) - 1))
        p1 = np.concatenate((pop[i][0:point], pop[i + 1][point:]))
        p2 = np.concatenate((pop[i + 1][0:point], pop[i][point:]))

        new_pop[i] = np.array(p1)
        new_pop[i + 1] = np.array(p2)

    for i in new_pop:
        if corr_method == 1:
            i = correct_solution(w, v, W, i)
        elif corr_method == 2:
            i = correct_solution2(w, v, W, i)
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
        elif corr_method == 2:
            pop[i] = correct_solution2(w, v, W, pop[i])
    return pop

def evolve_knapsack(w, v, W, pop_size, pxover, pmutate, generations, corr_method):
    pop = gen_pop(w, v, W, pop_size, corr_method)
    evals = evaluate(pop, v)
    i = np.argmax(evals)
    best = pop[i].copy()
    best_V = evals[i]
    best_iter = 0
    v_all = [best_V]
    v_best = [best_V]
    v_mean = [np.mean(evals)]

    # print('initial best', best_V)

    for i in range(generations):
        pop = select(pop, evals)
        pop = xover(pop, pxover, w, v, W, corr_method)
        pop = mutate(pop, pmutate, w, v, W, corr_method)
        evals = evaluate(pop, v)
        ii = np.argmax(evals)
        temp_best_v = evals[ii]
        if temp_best_v > best_V:
            best_V = temp_best_v
            best_iter = i + 1
            best = pop[ii].copy()
            print('better solution of ', best_V, 'in', best_iter)
        v_all.append(temp_best_v)
        v_best.append(best_V)
        v_mean.append(np.mean(evals))

    return best, w[best].sum(), best_V


def genetic_test(number_of_tests, w, v, W, pop_size, pxover, pmutate, generations, corr_method):
    w_best = np.ndarray(shape=(number_of_tests,))
    v_best = np.ndarray(shape=(number_of_tests,))
    p_best = np.ndarray(shape=(number_of_tests, len(w)), dtype="bool")

    for t in range(number_of_tests):
        best, best_w, best_v = evolve_knapsack(w, v, W, pop_size, pxover, pmutate, generations, corr_method)
        p_best[t] = best
        w_best[t] = best_w
        v_best[t] = best_v

    max = np.argmax(v_best)
    best_pop = p_best[max]
    return np.average(w_best), np.average(v_best), best_pop


def genetic_test_complete(knapsack_size):
    # Setting custom seed
    np.random.seed(125689)

    w, v = generate_problem(1, 100, 1, 100, knapsack_size)  # w - wagi, v - wartosci
    kn_weight = w.sum()
    kn_value = v.sum()
    W = int(0.5 * kn_weight)
    print("Rozmiar problemu:", knapsack_size)

    print('\tPojemnosc:', W)
    print('\tWagi:', w)
    print("\tSuma wag:", kn_weight)
    print('\tWartosci:', v)
    print("\tSuma wartosci:", kn_value)

    # Setting seed back to random
    np.random.seed(int(time.time()))

    pop_size = 50
    pxover = 0.9
    pmutate = 0.02
    generations = 100
    number_of_tests = 20

    print("\n\nAlgorytm genetyczny")

    # Basic correction method
    w_basic, v_basic, best_pop_basic = genetic_test(number_of_tests, w, v, W, pop_size, pxover, pmutate, generations, 1)
    print("Podstawowa metoda naprawcza:")
    np.savetxt('Genetic - basic correction method, size ' + str(knapsack_size) + ".txt", best_pop_basic, delimiter=',')

    print("\tSrednia waga:", w_basic)
    print("\tSrednia wartosc:", v_basic)

    # Custom correction method
    w_custom, v_custom, best_pop_custom = genetic_test(number_of_tests, w, v, W, pop_size, pxover, pmutate, generations, 2)
    print("Ulepszona metoda naprawcza:")
    np.savetxt('Genetic - custion correction method, size ' + str(knapsack_size) + ".out", best_pop_custom, delimiter=',')

    print("\tSrednia waga:", w_custom)
    print("\tSrednia wartosc:", v_custom)


def heuristics_tests(knapsack_size):
    # Setting custom seed
    np.random.seed(125689)

    w, v = generate_problem(1, 100, 1, 100, knapsack_size)  # w - wagi, v - wartosci
    kn_weight = w.sum()
    kn_value = v.sum()
    W = int(0.5 * kn_weight)
    print("Rozmiar problemu:", knapsack_size)

    number_of_tests = 20

    # Setting seed back to random
    np.random.seed(int(time.time()))

    print("Random search:")
    w_best = np.ndarray(shape=(number_of_tests,), dtype="float")
    v_best = np.ndarray(shape=(number_of_tests,), dtype="float")
    for i in range(number_of_tests):
        sol_random_search = search_random(w, v, W, 1000)
        w_best[i] = sol_random_search[1]
        v_best[i] = sol_random_search[2]

    print('\tSrednia waga:', np.average(w_best))
    print('\tSrednia wartosc:', np.average(v_best))

    print("Greedy search:")
    w_best = np.ndarray(shape=(number_of_tests,), dtype="float")
    v_best = np.ndarray(shape=(number_of_tests,), dtype="float")
    for i in range(number_of_tests):
        sol_greedy = search_greedy_improvement(w, v, W, 1000)
        w_best[i] = sol_greedy[1]
        v_best[i] = sol_greedy[2]

    print('\tSrednia waga:', np.average(w_best))
    print('\tSrednia wartosc:', np.average(v_best))

    print("Value first:")
    w_best = np.ndarray(shape=(number_of_tests,), dtype="float")
    v_best = np.ndarray(shape=(number_of_tests,), dtype="float")
    for i in range(number_of_tests):
        sol_value_first = get_value_first(w, v, W)
        w_best[i] = sol_value_first[1]
        v_best[i] = sol_value_first[2]

    print('\tSrednia waga:', np.average(w_best))
    print('\tSrednia wartosc:', np.average(v_best))

    print("Ratio first:")
    w_best = np.ndarray(shape=(number_of_tests,), dtype="float")
    v_best = np.ndarray(shape=(number_of_tests,), dtype="float")
    for i in range(number_of_tests):
        sol_ratio_first = get_ratio_first(w, v, W)
        w_best[i] = sol_ratio_first[1]
        v_best[i] = sol_ratio_first[2]

    print('\tSrednia waga:', np.average(w_best))
    print('\tSrednia wartosc:', np.average(v_best))
    print("\n")

def tests():
    genetic_test_complete(50)
    heuristics_tests(50)
    genetic_test_complete(100)
    heuristics_tests(100)
    genetic_test_complete(300)
    heuristics_tests(300)

tests()