import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nbits(a, b, dx):
    length = abs(b - a)/dx
    B = math.ceil(math.log(length, 2))

    tmp = 2**B
    dx_new = abs(b-a)/tmp

    return B, dx_new


def gen_population(P, N, B):
    population = np.ndarray(shape=(N, P * B), dtype="int")
    for i in range(P):
        for j in range(B * N):
            population[i][j] = np.random.randint(0, 2)


    return population


def decode_individual(individual, N, B, a, dx):
    decoded = np.ndarray(shape=(N,))

    for i in range(N):
        decimal = 0
        for j in range(B):
            decimal += individual[i * B + j] * 2**(B - j - 1)
        decoded[i] = a + (decimal * dx)
    return decoded


def evaluate_population(func, pop, N, B, a, dx):
    evaluated_pop = np.array([func(decode_individual(i, N, B, a, dx)) for i in pop])
    return evaluated_pop

def get_best(pop, evaluated_pop):
    best_value = np.amax(evaluated_pop)
    best_individual = np.array(pop[np.argmax(evaluated_pop)])
    return best_individual, best_value


def roulette(pop, evaluated_pop):
    if evaluated_pop.min() < 1:
        evaluated_pop += math.fabs(evaluated_pop.min()) + 1

    evaluated_pop = np.cumsum(evaluated_pop / evaluated_pop.sum())
    new_pop = np.ndarray(shape=np.shape(pop), dtype=np.float64)

    for i in range(len(pop)):
        j = 0
        r = np.random.random_sample()
        while evaluated_pop[j] < r:
            j += 1
        new_pop[i] = pop[j]

    return new_pop


def cross(pop, pk):
    new_pop = np.ndarray(shape=(len(pop), len(pop[0])), dtype="int")
    for i in range(0, len(pop) - 1, 2):
        if np.random.random() < pk:
            cross_p = len(pop[0]/2)
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


def mutate(pop, pm):
    new_pop = np.array([[not (x) if np.random.random_sample() < pm else x for x in pop[i]] for i in range(len(pop))])
    return new_pop

def obj_func(x):
    w1 = 1 + (x[0]-1)/4
    w2 = 1 + (x[1] - 1) / 4

    sum = (w1 - 1)**2 * (1 + 10 * math.sin(math.pi * w1 + 1)**2)

    return math.sin(math.pi * w1)**2 + sum + (w2 - 1)**2 * (1 + math.sin(2 * math.pi * w2)**2)


def gen_alg_implementation(fun, pop_size, pk, pm, generations, dx):
    N = 2
    B, dx = nbits(-10, 10, dx)
    pop = gen_population(pop_size, B=B)
    best_generation = 1
    list_best = []
    list_best_generation = []
    list_mean = []
    evaluated_pop = evaluate_population(fun, pop, dx, B=B)
    best_sol = get_best(pop, evaluated_pop)

    first_pop_sol = np.apply_along_axis(decode_individual, 1, pop, dx, -10, N, B)
    first_pop_eval = np.copy(evaluated_pop)

    for i in range(2, generations + 1):
        pop = roulette(pop, evaluated_pop)
        pop = cross(pop, pk)
        pop = mutate(pop, pm)

        evaluated_pop = evaluate_population(fun, pop, dx, B=B)
        best_sol_temp = get_best(pop, evaluated_pop)

        list_best_generation.append(best_sol_temp[1])
        list_mean.append(np.mean(evaluated_pop))

        if best_sol_temp[1] > best_sol[1]:
            best_sol = best_sol_temp
            best_generation = i

        list_best.append(best_sol[1])

        if i == generations // 2:
            mid_pop_sol = np.apply_along_axis(decode_individual, 1, pop, dx, -10, N, B)
            mid_pop_eval = np.copy(evaluated_pop)

        if i == generations:
            last_pop_sol = np.apply_along_axis(decode_individual, 1, pop, dx, -10, N, B)
            last_pop_eval = np.copy(evaluated_pop)

        best_x, best_y = decode_individual(best_sol[0], dx, -10, N, B)
        best_z = best_sol[1]

        fig = plt.figure()
        plt.suptitle('Wykresy pierwszej, środkowej i ostatniej populacji. Najlepszy osobnik zaznaczony (*)')
        a1 = fig.add_subplot(131, projection='3d')
        x = first_pop_sol.T[0]
        y = first_pop_sol.T[1]
        z = first_pop_eval
        a1.scatter(x, y, z, c=z, marker='o')
        a1.scatter(best_x, best_y, best_z, c='r', s=30, marker='*')
        plt.title('First population')
        plt.axis([-10, 10, -10, 10])
        a1.set_xlabel('x')
        a1.set_ylabel('y')
        a1.set_zlabel('z')
        a1.view_init(10, 40)

        a2 = fig.add_subplot(132, projection='3d')
        x = mid_pop_sol.T[0]
        y = mid_pop_sol.T[1]
        z = mid_pop_eval
        a2.scatter(x, y, z, c=z, marker='o')
        a2.scatter(best_x, best_y, best_z, c='r', s=30, marker='*')
        plt.title('Middle population')
        plt.axis([-10, 10, -10, 10])
        a2.set_xlabel('x')
        a2.set_ylabel('y')
        a2.set_zlabel('z')
        a2.view_init(10, 40)

        a3 = fig.add_subplot(133, projection='3d')
        x = last_pop_sol.T[0]
        y = last_pop_sol.T[1]
        z = last_pop_eval
        a3.scatter(x, y, z, c=z, marker='o')
        a3.scatter(best_x, best_y, best_z, c='r', s=30, marker='*')
        plt.title('Last population')
        plt.axis([-10, 10, -10, 10])
        a3.set_xlabel('x')
        a3.set_ylabel('y')
        a3.set_zlabel('z')
        a3.view_init(10, 40)

        plt.show()

    return best_sol, best_generation, list_best, list_best_generation, list_mean