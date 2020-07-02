from utils import *
from pop_sim import simulate
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import _pickle as pickle
np.random.seed(11523)
number_of_the_genes = 1000
causal_genes = number_of_the_genes
length = 30
vg0 = np.eye(number_of_the_genes)/number_of_the_genes
mu = 0.5
c = 1 - np.eye(number_of_the_genes)
p = 0.5
ve = 1
a = np.random.normal(size=(1, number_of_the_genes))
b = mu*a
pop_size = 10000
number_of_sims = 30

expected_vg, expected_vy, expected_m, expected_h2, expected_sib_cov = simulate_generations(length, vg0, c, a, b, p, ve)
expected_vy = expected_vy[1:]
results = []
for i in range(number_of_sims):
    print(i, "th sim")
    result = simulate(pop_size, length, number_of_the_genes, causal_genes, mu, a, vg0, p, ve, True)
    results.append(result["vy"][1:])

data = {
    "expected": expected_vy,
    "result": results,
    "value": "vy"
}
with open(f"comparison_data_vy_p={p}_mu={mu}.pickle", "wb") as f:
    pickle.dump(data, f)

plt.plot(range(29), np.mean(results, axis = 0), color = "b", label="sim result")
plt.plot(range(29), expected_vy, color = "r", label="expected")
plt.xlabel("Vy")
plt.ylabel("generation")
plt.title(f"Average of Vys over 1000 simulations \n vs expected\n p={p}, mu={mu}")
plt.legend()
plt.savefig(f"comparison_sim_exp_p={p}_mu={mu}")
