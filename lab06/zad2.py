import pymc3 as pm

with pm.Model() as model:
    # variables telling about possibility of burglary and earthquake
    burglary = pm.Bernoulli('burglary', .001)
    earthquake = pm.Bernoulli('earthquake', .002)

    # Probability of alarm
    p_alarm = pm.Deterministic('p_alarm', pm.math.switch(burglary, pm.math.switch(earthquake, .95, .94),
                                                         pm.math.switch(earthquake, .29, .001)))
    alarm = pm.Bernoulli('alarm', p_alarm)

    # Probability of John calling during alarm
    p_john = pm.Deterministic('p_john', pm.math.switch(alarm, .9, .05))
    john = pm.Bernoulli('john', p_john)

    p_mary = pm.Deterministic('p_mary', pm.math.switch(alarm, .7, .01))
    mary = pm.Bernoulli('mary', p_mary)


with model:
    trace = pm.sample(20000, chains=1)

    # print("{:<50}".format("Fakty:"), "Prawdopodobieństwo popełnienia przestępstwa")

    # 1. Probability of alarm
    p_alarm = (trace['alarm'].sum()/len(trace['alarm']))
    print("1. Prawdopodobieństwo wystąpienia alarmu:", p_alarm)

    # 2. Probability of burglary under condition of alarm
    p_burglary_alarm = (trace['burglary'] * trace['alarm']).sum() / trace['alarm'].sum()
    print("2. Prawdopodobieństwo włamania jeśli włączył się alarm:", p_burglary_alarm)

    # 3. Probability of earthquake if alarm
    p_earthquake_alarm = (trace['earthquake'] * trace['alarm']).sum() / trace['alarm'].sum()
    print("3. Prawdopodobieństwo trzęsienia ziemi jeśli włączył się alarm:", p_earthquake_alarm)

    # 4. Probability of someone calling during burglary TODO This solution is incorrect
    p_burglary_call = (p_burglary_alarm * (trace['john'] * trace['mary']).sum()) / (trace['john'] * trace['mary']).sum()
    print("4. Prawdopodobieństwo że ktoś zadzwoni w czasie włamania:", p_burglary_call)

