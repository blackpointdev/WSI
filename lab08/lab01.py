import math
import numpy as np


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


