import numpy as np


class Perceptron:
    # Inicjalizator, ustawiający atrybut self.w jako wektor losowych wag, n ilość sygnałów wejściowych
    def __init__(self, n):
        self.w = np.random.randint(-1, 2, size=(1, n))
        print(self.w)

    # Metoda obliczająca odpowiedz modelu dla zadanego sygnału wejściowego x=[1,x1,x2,...,xN]
    def predict(self, x):
        sum = 0
        for i in range(len(x)):
            sum += x[i] + self.w[i]

        if sum > 0:
            return True
        else:
            return False

    # Metoda uczenia według reguły perceptronu, xx - zbiór danych uczących, d - odpowiedzi,
    # eta - współczynnik uczenia,
    # tol - tolerancja (czyli jak duży błąd jesteśmy w stanie zaakceptować)
    def train(self, xx, d, eta, tol):
        # YOUR CODE HERE
        raise NotImplementedError()

    # Metoda obliczająca błąd dla danych testowych xx
    # zwraca błąd oraz wektor odpowiedzi perceptronu dla danych testowych
    def evaluate_test(self, xx, d):
        # YOUR CODE HERE
        raise NotImplementedError()

Perceptron(5)