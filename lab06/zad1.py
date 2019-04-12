import pymc3 as pm
# import numpy as np
# import matplotlib.pyplot as plt

with pm.Model() as Bayes_model:
    # zmienna mówiąca czy podejrzany jest winny, o rozkładzie Bernoulliego z prawd. 0.5
    zabil = pm.Bernoulli('zabil', 0.5)

    # prawdopodobieństwo znalezienia odcisków palców podejrzanego pod warunkiem że zabił
    p_odciski = pm.Deterministic("p_odciski", pm.math.switch(zabil, 0.7, 0.3))
    odciski = pm.Bernoulli("odciski", p_odciski)

    p_alibi = pm.Deterministic("p_alibi", pm.math.switch(zabil, 0.8, 0.4))
    alibi = pm.Bernoulli("alibi", p_alibi)

    p_motyw = pm.Deterministic("p_motyw", pm.math.switch(zabil, 0.9, 0.5))
    motyw = pm.Bernoulli("motyw", p_alibi)

    p_widziany = pm.Deterministic("p_widziany", pm.math.switch(zabil, 0.4, 0.2))
    widziany = pm.Bernoulli("widziany", p_alibi)

    p_rysopis = pm.Deterministic("p_rysopis", pm.math.switch(zabil, 0.2, 0.4))
    rysopis = pm.Bernoulli("rysopis", p_rysopis)

with Bayes_model:
    trace = pm.sample(20000, chains=1)

print("{:<50}".format("Fakty:"), "Prawdopodobieństwo popełnienia przestępstwa")

# prawdopodobieństwo popełnienia zbrodni pod warunkiem że znaleziono odciski palcow
p_zabil_odciski = (trace['zabil']*trace['odciski']).sum()/trace['odciski'].sum()
print("{:<50}".format("- znaleziono odciski podejrzanego:"), p_zabil_odciski)

# prawdopodobieństwo popełnienia zbrodni pod warunkiem że podejrzany nie ma alibi i miał motyw
p_alibi_motyw = (trace['zabil'] * trace['alibi'] * trace['motyw']).sum() / (trace['alibi'] * trace['motyw']).sum()
print("{:<50}".format("- podejrzany miał motyw i nie miał alibi:"), p_alibi_motyw)

# prawdopodobieństwo popełnienia zbrodni pod warunkiem że znaleziono odciski palców, podejrzany był widziany
# w okolicy miejsca zamierszkania handlarza bronią, ale otrzymany rysopis zabójcy nie pasuje do podejrzanego
p_odciski_widziany_rysopis = (trace['zabil'] * trace['odciski'] * trace['widziany'] * trace['rysopis']).sum() / \
                             (trace['odciski'] * trace['widziany'] * trace['rysopis']).sum()
print("{:<50}".format("- znaleziono odciski palców podejrzanego,\n  podejrzany był widziany w okolicy miejsca\n"
      "  zamierszkania handlarza bronią, ale otrzymany"))
print("{:<50}".format("  rysopis zabójcy nie pasuje do podejrzanego"), p_odciski_widziany_rysopis)

