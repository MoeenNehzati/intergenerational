from utils import *
import seaborn as sns
from matplotlib import pyplot as plt
np.random.seed(11523)
number_of_the_genes = 1000
length = 30
vg0 = np.eye(number_of_the_genes)/number_of_the_genes
c = 0.5 - 0.5*np.eye(number_of_the_genes)
a = np.random.normal(size = (1, number_of_the_genes))
p = 0.5
mu = 0.5
ve = 1

def simulate_and_plot(length, vg0, c, a, p, mu, ve):
    plt.clf()
    vg, vy, m = simulate_generations(length, vg0, c, a, p, mu, ve)
    title = f'Vy with mu={mu}, p={p}, Ve={ve}, number_of_the_genes={number_of_the_genes},\n Vg0=I{number_of_the_genes}^-1, A~N'
    fig = sns.lineplot("generation", "Vy", data=pd.DataFrame({"Vy":vy, "generation":range(length)}))
    fig.set_title(title)
    fig.get_figure().savefig("../out/figs/"+title.replace(" ", "").replace("\n", "")+".png")
    plt.clf()
    title = f'h2 with mu={mu}, p={p}, Ve={ve}, number_of_the_genes={number_of_the_genes},\n Vg0=I{number_of_the_genes}^-1, A~N'
    fig = sns.lineplot("generation", "h2", data=pd.DataFrame({"h2":[(a@vg[i]@a.T).item()/vy[i] for i in range(length)], "generation":range(length)}))
    fig.set_title(title)
    fig.get_figure().savefig("../out/figs/"+title.replace(" ", "").replace("\n", "")+".png")
    plt.clf()

simulate_and_plot(length, vg0, c, a, p, 0.25, ve)
simulate_and_plot(length, vg0, c, a, p, 0.5, ve)
simulate_and_plot(length, vg0, c, a, p, 1, ve)
simulate_and_plot(length, vg0, c, a, p, 2, ve)

