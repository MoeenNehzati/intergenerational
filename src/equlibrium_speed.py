import numpy as np 
from tqdm import tqdm
from numpy import sqrt 
from scipy.stats import multivariate_normal
from utils import simulate_generations
from matplotlib import pyplot as plt
import pickle
length = 30
number_of_the_genes = 2000
maf = 0.001
vg0 = np.eye(number_of_the_genes)*(2*maf*(1-maf))
p = 0.5
ve = 1
va = 1
pop_size = 20000
c = 1 - np.eye(number_of_the_genes)

a = np.random.normal(size=(1, number_of_the_genes))
sample_g = np.random.binomial(2, maf, (number_of_the_genes, pop_size//2))
result_var = np.var(a@sample_g)
va = 1/result_var
a = np.random.normal(scale=np.sqrt(va), size=(1, number_of_the_genes))
sample_g = np.random.binomial(2, maf, (number_of_the_genes, pop_size//2))
result_var = np.var(a@sample_g)
print("AAAAAAAAAAAAAAAAAAAAAAAA")
print(result_var)



data = {}
data["length"] = length
data["vg0"] = vg0
data["c"] = c
data["p"] = p
data["va"] = va
data["ve"] = ve
data["key"] = "vb, corr_ab"
data["value"] = "vysi, hi2, sib_corri"
data["data"] = {}

vb = 0
cor_ab = 0

ab_cov = cor_ab * np.sqrt(va*va*vb)
ab_cov_matrix = [[va, ab_cov], [ab_cov, va*vb]]
ab = multivariate_normal.rvs(cov = ab_cov_matrix, size=number_of_the_genes)
a = ab[:,0].reshape((1, -1))
b = ab[:,1].reshape((1, -1))
vg, vy, m, h2, sib_cov = simulate_generations(length, vg0, c, a, b, p, ve)
data["data"][(vb, cor_ab)] = vy, h2, sib_cov
proportional_change_vy = (np.array(vy[1:])-np.array(vy[:-1]))/np.array(vy[:-1])
plt.plot(range(5, length), proportional_change_vy[4:], label = f"vb={vb}, cor_ab={cor_ab}", linewidth = 2)

for vb in [0.25, 0.5, 1]:    
    for cor_ab in [0, 0.5, 1]:
        print(vb, cor_ab)
        ab_cov = cor_ab * np.sqrt(va*va*vb)
        ab_cov_matrix = [[va, ab_cov], [ab_cov, va*vb]]
        ab = multivariate_normal.rvs(cov = ab_cov_matrix, size=number_of_the_genes)
        a = ab[:,0].reshape((1, -1))
        b = ab[:,1].reshape((1, -1))
        vg, vy, m, h2, sib_cov = simulate_generations(length, vg0, c, a, b, p, ve)
        data["data"][(vb, cor_ab)] = vy, h2, sib_cov
        proportional_change_vy = (np.array(vy[1:])-np.array(vy[:-1]))/np.array(vy[:-1])
        plt.plot(range(5, length), proportional_change_vy[4:], label = f"vb={vb}, cor_ab={cor_ab}", linewidth = 2)



plt.legend()
plt.xlabel('generation')
plt.ylabel('change in Vy')
title = f'delta Vy With MAF={maf}, p={p}, Ve={ve}, number_of_the_genes={number_of_the_genes},\n Vg0=I{number_of_the_genes}*MAF^{-1}'
plt.title(title)
plt.savefig(title.replace(" ", "").replace("\n", "") + ".png")
plt.clf()

with open("formula_data_" + title.replace(" ", "").replace("\n", "") + ".pickle", "wb") as f:
    pickle.dump(data, f)


plt.clf()
with open("formula_data_" + title.replace(" ", "").replace("\n", "") + ".pickle", "rb") as f:
    data = pickle.load(f)

plt.clf()
for key, val in data["data"].items():
    print("AA")
    vb, cor_ab = key
    vy, h2, sib_cov = val
    proportional_change_h2 = (np.array(h2[1:])-np.array(h2[:-1]))/np.array(h2[:-1])
    plt.plot(range(5, length), proportional_change_h2[4:], label = f"vb={vb}, cor_ab={cor_ab}", linewidth = 2)
    plt.legend()
plt.xlabel('generation')
plt.ylabel('change in h2')
title = f'delta h2 With MAF={maf}, p={p}, Ve={ve}, number_of_the_genes={number_of_the_genes},\n Vg0=I{number_of_the_genes}*MAF^{-1}'
plt.title(title)
plt.savefig(title.replace(" ", "").replace("\n", "") + ".png")



plt.clf()
for key, val in data["data"].items():
    vb, cor_ab = key
    vy, h2, sib_cov = val
    # proportional_change_h2 = (np.array(h2[1:])-np.array(h2[:-1]))/np.array(h2[:-1])
    plt.plot(range(length), h2, label = f"vb={vb}, cor_ab={cor_ab}", linewidth = 2)
    plt.legend()
plt.xlabel('generation')
plt.ylabel('h2')
title = f'h2 With MAF={maf}, p={p}, Ve={ve}, number_of_the_genes={number_of_the_genes},\n Vg0=I{number_of_the_genes}*MAF^{-1}'
plt.title(title)
plt.savefig(title.replace(" ", "").replace("\n", "") + ".png")


plt.clf()
for key, val in data["data"].items():
    vb, cor_ab = key
    vy, h2, sib_cov = val
    # proportional_change_h2 = (np.array(h2[1:])-np.array(h2[:-1]))/np.array(h2[:-1])
    plt.plot(range(length), vy, label = f"vb={vb}, cor_ab={cor_ab}", linewidth = 2)
    plt.legend()
plt.xlabel('generation')
plt.ylabel('vy')
title = f'vy With MAF={maf}, p={p}, Ve={ve}, number_of_the_genes={number_of_the_genes},\n Vg0=I{number_of_the_genes}*MAF^{-1}'
plt.title(title)
plt.savefig(title.replace(" ", "").replace("\n", "") + ".png")