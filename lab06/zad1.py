import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

with pm.Model() as Bayes_model:
    # zmienna mówiąca czy podejrzany jest winny, o rozkładzie Bernoulliego z prawd. 0.5
    zabil = pm.Bernoulli('zabil', 0.5)

    # prawdopodobieństwo znalezienia odcisków palców podejrzanego pod warunkiem że zabił
    p_odciski = pm.Deterministic("p_odciski", pm.math.switch(0.7, 0.3))
    odciski = pm.Bernoulli("odciski", p_odciski)

    p_alibi = pm.Deterministic("p_alibi", pm.math.switch(0.8, 0.4))
    alibi = pm.Bernoulli("alibi", p_alibi)

    p_motyw = pm.Deterministic("p_motyw", pm.math.switch(0.9, 0.5))
    motyw = pm.Bernoulli("motyw", p_alibi)

    p_alibi = pm.Deterministic("p_alibi", pm.math.switch(0.8, 0.4))
    alibi = pm.Bernoulli("alibi", p_alibi)
