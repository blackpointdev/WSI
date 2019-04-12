import pymc3 as pm

with pm.Model() as model:
    # variables telling about possibility of burglary and earthquake
    burglary = pm.Bernoulli('burglary', .001)
    earthquake = pm.Bernoulli('earthquake', .002)

    # Probability of alarm
    p_alarm = pm.Deterministic('p_alarm', pm.math.switch(burglary, pm.math.switch(earthquake, .95, .94),
                                                         pm.math.switch(earthquake, .29, .001)))
    alarm = pm.Bernoulli('alarm', p_alarm)
    # probability of alarm in case if burglary and earthquake

with model:
    trace = pm.sample(20000, chains=1)

    # print("{:<50}".format("Fakty:"), "Prawdopodobieństwo popełnienia przestępstwa")
    # probability of alarm
    p_alarm = (trace['alarm'].sum()/len(trace['alarm']))
    print("Prawdopodobieństwo wystąpienia alarmu:", p_alarm)

    # Probability of burglary under condition of alarm
    p_burglary_alarm = (.001 * .94 / p_alarm) # TODO Find correct solution, because this sucks
    print("Prawdopodobieństwo włamania jeśli włączył się alarm:", p_burglary_alarm)
