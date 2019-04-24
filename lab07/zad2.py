from fuzzython.fsets.triangular import Triangular
from fuzzython.variable import Variable
from fuzzython.adjective import Adjective
from fuzzython.ruleblock import RuleBlock
import numpy as np
import matplotlib.pyplot as plt
from fuzzython.systems.sugeno import SugenoSystem
from mpl_toolkits.mplot3d import Axes3D

# pomocnicza funkcja do rysowania zbiorów rozmytych
def plot_fuzzyset(ax, fuzzy_set, x, *args, **kwargs):
    y = np.array([fuzzy_set(e) for e in x])
    ax.plot(x, y,  *args, **kwargs)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()


p_y_low = Triangular((1, 1), (10, 0), (20, 0))
p_y_medium = Triangular((1, 0), (10, 1), (20, 0))
p_y_high = Triangular((1, 0), (10, 0), (20, 1))

a_p_y_low = Adjective("p_y_low", p_y_low)
a_p_y_medium = Adjective("p_y_medium", p_y_medium)
a_p_y_high = Adjective("p_y_high", p_y_high)

yesterday = Variable("yesterday", "$", a_p_y_low, a_p_y_medium, a_p_y_high)

p_t_low = Triangular((1, 1), (10, 0), (20, 0))
p_t_medium = Triangular((1, 0), (10, 1), (20, 0))
p_t_high = Triangular((1, 0), (10, 0), (20, 1))

a_p_t_low = Adjective("p_t_low", p_t_low)
a_p_t_medium = Adjective("p_t_medium", p_t_medium)
a_p_t_high = Adjective("p_t_high", p_t_high)

today = Variable("today", "$", a_p_t_low, a_p_t_medium, a_p_t_high)

p_tom_low = Triangular((1, 1), (10, 0), (20, 0))
p_tom_medium = Triangular((1, 0), (10, 1), (20, 0))
p_tom_high = Triangular((1, 0), (10, 0), (20, 1))

a_p_tom_low = Adjective("p_tom_low", p_tom_low)
a_p_tom_medium = Adjective("p_tom_medium", p_tom_medium)
a_p_tom_high = Adjective("p_tom_high", p_tom_high)

tomorrow = Variable('tomorrow', "$", a_p_tom_low, a_p_tom_medium, a_p_tom_high, defuzzification='COG', default=0)

# wykresy
x = np.linspace(0, 20, 1000)

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12,8))
((ax1), (ax2), (ax3)) = axs
plot_fuzzyset(ax1, p_y_low, x, 'b', label='p_y_low')
plot_fuzzyset(ax1, p_y_medium, x, 'g', label='p_y_medium')
plot_fuzzyset(ax1, p_y_high, x, 'r', label='p_y_high')
plot_fuzzyset(ax2, p_t_low, x, 'b', label='p_t_low')
plot_fuzzyset(ax2, p_t_medium, x, 'g', label='p_t_medium')
plot_fuzzyset(ax2, p_t_high, x, 'r', label='p_t_high')
plot_fuzzyset(ax3, p_tom_low, x, 'b', label='p_tom_low')
plot_fuzzyset(ax3, p_tom_medium, x, 'g', label='p_tom_medium')
plot_fuzzyset(ax3, p_tom_high, x, 'r', label='p_tom_high')
plt.show()

scope = locals()

rule1 = "if yesterday is a_p_y_low or today is a_p_t_low then z=yesterday*0.4+today*0.4"
rule2 = "if yesterday is a_p_y_high or today is a_p_t_high then z=yesterday*0.7+today*0.7"
rule3 = "if yesterday is a_p_y_medium and today is a_p_t_low or today is a_p_t_medium or today is a_p_t_high then " \
        "z=yesterday*0.3+today*0.7"
rule4 = "if today is a_p_t_medium and yesterday is a_p_y_low or yesterday is a_p_y_medium or yesterday is a_p_y_high " \
        "then z=yesterday*0.7+today*0.3"

block = RuleBlock('rb_takagi', operators=('MIN', 'MAX', 'ZADEH'), activation='MIN', accumulation='MAX')
block.add_rules(rule1, rule2, rule3, rule4, scope=scope)

sugeno = SugenoSystem('model_takagi', block)

# dane wejściowe
inputs = {'yesterday': 19, 'today': 19}

res = sugeno.compute(inputs)

print("Cena wczoraj:", inputs["yesterday"], ", cena dzisiaj:", inputs["today"])
print(res)

# %matplotlib notebook

# przygotowanie siatki
sampled = np.linspace(0, 20, 20)
x, y = np.meshgrid(sampled, sampled)
z = np.zeros((len(sampled), len(sampled)))

for i in range(len(sampled)):
    for j in range(len(sampled)):
        inputs = {'yesterday': x[i, j], 'today': y[i, j]}
        res = sugeno.compute(inputs)
        z[i, j] = res['rb_takagi']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
cset = ax.contourf(x, y, z, zdir='z', offset=-1, cmap='viridis', alpha=0.5)
cset = ax.contourf(x, y, z, zdir='x', offset=11, cmap='viridis', alpha=0.5)
cset = ax.contourf(x, y, z, zdir='y', offset=11, cmap='viridis', alpha=0.5)
ax.set_xlabel('yesterday')
ax.set_ylabel('today')
ax.set_zlabel('tommorow')
ax.view_init(30, 200)

