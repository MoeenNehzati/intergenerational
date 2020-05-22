from utils import *
np.random.seed(11523)
number_of_the_genes = 1000
length = 660
vg0 = np.eye(number_of_the_genes)/number_of_the_genes
c = 0.5 - 0.5*np.eye(number_of_the_genes)
a = np.random.normal(size = (1, number_of_the_genes))
p = 0.5
mu = 0.5
ve = 1
vg, vy, m = simulate_generations(length, vg0, c, a, p, mu, ve)
vy
np.array(vy[1:]).reshape((1,-1)) - np.array(vy[:-1]).reshape((1,-1))
import seaborn as sns
title = f'With mu={mu}, p={p}, Ve={ve}, number_of_the_genes={number_of_the_genes},\n Vg0=I{number_of_the_genes}^-1, A~N'
fig = sns.lineplot("generation", "Vy", data=pd.DataFrame({"Vy":vy, "generation":range(length)}))
fig.set_title(title)
fig.get_figure().savefig("../out/figs/"+title.replace(" ", "").replace("\n", "")+".png")
from matplotlib import pyplot as plt
plt.show()