import numpy as np
import matplotlib.pyplot as plt
#%matplotlib notebook

#wmin - minimalna waga przedmiotu
#wmax - maksymalna waga przedmiotu
#vmin - minimalna wartość przedmiotu
#vmax - maksymalna wartość przedmiotu
#items_num - liczba dostępnych przedmiotów
def generate_problem(wmin, wmax, vmin, vmax, items_num):
    w = np.random.randint(wmin, wmax, size=items_num)  #weight
    v = np.random.randint(vmin, vmax, size=items_num)  #values
    return w, v

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

def get_random_solution_improved(w, v, W):
    num = w.shape[0]
    sol = np.random.randint(0,2, size=num, dtype=np.bool)  # 1 / True oznacza, ze przedmiot jest wybrany
    _V = np.sum(v[sol])
    _W = np.sum(w[sol])
    if _W > W:
        correct_solution2(w,v,W,sol)
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
    # plt.figure()
    # plt.plot(v_all)
    # plt.plot(v_best)
    # plt.show()
    return best_sol, best_W, best_V, v_all, v_best

def search_random_improved(w,v,W,iters):
    best_sol, best_W, best_V = get_random_solution_improved(w,v,W)
    v_all = [best_V]
    v_best = [best_V]
    for i in range(iters):
        sol, _W, _V = get_random_solution_improved(w,v,W)
        if best_V < _V:
            best_sol, best_W, best_V = sol, _W, _V
        v_all.append(_V)
        v_best.append(best_V)
    # plt.figure()
    # plt.plot(v_all)
    # plt.plot(v_best)
    # plt.show()
    return best_sol, best_W, best_V, v_all, v_best

def search_greedy(w, v, W, iters):
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
    # plt.figure()
    # plt.plot(v_all)
    # plt.plot(v_best)
    # plt.show()
    return best_sol, best_W, best_V, v_all, v_best

def search_greedy_improved(w, v, W, iters):
    best_sol, best_W, best_V = get_random_solution_improved(w,v,W)
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
            correct_solution2(w,v,W,sol)
        _V = v[sol].sum()
        _W = w[sol].sum()
        if best_V < _V:
            best_sol, best_W, best_V = sol.copy(), _W, _V
        v_all.append(_V)
        v_best.append(best_V)
    # plt.figure()
    # plt.plot(v_all)
    # plt.plot(v_best)
    # plt.show()
    return best_sol, best_W, best_V, v_all, v_best

num = 50  # liczba przedmiotów
wmin = 1  # minimlana waga
wmax = 100  # maksymalna waga
vmin = 1  # minimalna wartosc
vmax = 100  # maksymalna wartosc
knapsack_perc = 0.5  # pojemnosc plecaka jako procent sumy wag wszystkich przedmiotow

w, v = generate_problem(wmin, wmax, vmin, vmax, num)  # w - wagi, v - wartosci
Wall = w.sum()
Vall = v.sum()
W = int(knapsack_perc * Wall) # pojemnosc plecaka

number_of_tests = 10

random_search = np.ndarray(shape=(number_of_tests, 2))
random_search_improved = np.ndarray(shape=(number_of_tests, 2))
greedy_search = np.ndarray(shape=(number_of_tests, 2))
greedy_search_improved = np.ndarray(shape=(number_of_tests, 2))

for i in range(number_of_tests):
    greedy = search_greedy(w, v, W, 1000)
    random = search_random(w, v, W, 1000)
    greedy_search[i] = [greedy[1], greedy[2]]
    random_search[i] = [random[1], random[2]]

    greedy = search_greedy_improved(w, v, W, 1000)
    random = search_random_improved(w, v, W, 1000)
    greedy_search_improved[i] = [greedy[1], greedy[2]]
    random_search_improved[i] = [random[1], random[2]]

print("Greedy search")
print('\tSuma wag:', np.average(a=greedy_search, axis=0)[0])
print('\tSuma wartosci:', np.average(a=greedy_search, axis=0)[1])

print("Greedy search improved")
print('\tSuma wag:', np.average(a=greedy_search_improved, axis=0)[0])
print('\tSuma wartosci:', np.average(a=greedy_search_improved, axis=0)[1])

print("Random search")
print('\tSuma wag:', np.average(a=random_search, axis=0)[0])
print('\tSuma wartosci:', np.average(a=random_search, axis=0)[1])

print("Random search improved")
print('\tSuma wag:', np.average(a=random_search_improved, axis=0)[0])
print('\tSuma wartosci:', np.average(a=random_search_improved, axis=0)[1])