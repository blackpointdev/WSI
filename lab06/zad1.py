import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

with pm.Model() as grypa_model:

    zabil = pm.Bernoulli('zabil', 0.5)

    # prawd. kataru pod warunkiem grypy
    # jeżeli grypa to p_katar=0.5, jeżeli nie grypa to p_katar=0.3
    p_katar = pm.Deterministic('p_katar', pm.math.switch(zabil, 0.5, 0.3))
    # zmienna katar o rozkładie Bernoulliego
    katar = pm.Bernoulli('katar', p_katar)

    # prawd. kaszlu pod warunkiem grypy
    # jeżeli grypa to p_kaszel=0.3, jeżeli nie grypa to p_kaszel=0.3
    p_kaszel = pm.Deterministic('p_kaszel', pm.math.switch(grypa, 0.3, 0.3))
    # zmienna kaszel o rozkładie Bernoulliego
    kaszel = pm.Bernoulli('kaszel', p_kaszel)

    # prawd. goraczki pod warunkiem grypy
    # jeżeli grypa to p_goraczka=0.8, jeżeli nie grypa to p_goraczka=0.4
    p_goraczka = pm.Deterministic('p_goraczka', pm.math.switch(grypa, 0.8, 0.4))
    goraczka = pm.Bernoulli('goraczka', p_goraczka)

with grypa_model:
    trace = pm.sample(20000, chains=1)  #liczba symulacji, są inne parametry warto sprawdzić w dokumentacji

# w module pymc3 dostępne są metody do wygernerowania różnego rodzaju wykresów np.
axs = pm.traceplot(trace, varnames=['grypa', 'katar', 'kaszel', 'goraczka'])
plt.show()
