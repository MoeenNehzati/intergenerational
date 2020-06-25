import numpy as np 
from tqdm import tqdm
from numpy import sqrt 
from scipy.stats import multivariate_normal

np.random.seed(11523)
pop_size = 10000
length = 30
number_of_the_genes = 2000
causal_genes = number_of_the_genes
vg0 = np.eye(number_of_the_genes)/number_of_the_genes
a = np.hstack((np.random.normal(size = (1, causal_genes)), np.array([[0]*(number_of_the_genes - causal_genes)])))
p = 0.5
mu = 0.5
ve = 1
f = 0
va = 1
for i in np.roots([2, -2, vg0[0, 0]]):
    if 0 < i <= 0.5:
        f = i
        break

if f == 0:
    raise Exception()

all_corrs = {}
for vb in [0, 0.25, 0.5, 0.75, 1]:
    for ab_cor in [0, 0.25, 0.5, 0.75, 1]:
        for p in [0.001, 0.25, 0.5, 0.75, 1]:
            print(vb, ab_cor, p)
            ab_cov = ab_cor * np.sqrt(va*vb)
            ab_cov = [[va, ab_cov], [ab_cov, vb]]
            ab = multivariate_normal.rvs(cov = ab_cov, size=number_of_the_genes)
            a = ab[:,0].reshape((1, -1))
            b = ab[:,1].reshape((1, -1))

            males = [None]*length
            females = [None]*length
            male_phenotypes = [None]*length
            female_phenotypes = [None]*length
            vy = [None]*length
            corrs = [None]*length
            corrs_original = [None]*length
            corrs_inbreeding = [None]*length
            corrs_causal = [None]*length
            deltas = [None]*length

            males[0] = np.random.binomial(2, f, (number_of_the_genes, pop_size//2))
            females[0] = np.random.binomial(2, f, (number_of_the_genes, pop_size//2))
            male_phenotypes[0] = a@males[0] + np.random.normal(0, sqrt(ve), (1, pop_size//2))
            female_phenotypes[0] = a@females[0] + np.random.normal(0, sqrt(ve), (1, pop_size//2))
            vy[0] = np.var(np.hstack((male_phenotypes[0][0], female_phenotypes[0][0])))

            for i in tqdm(range(1, length)):
                father_phenotype = male_phenotypes[i-1]
                mother_phenotype = female_phenotypes[i-1]
                father = males[i-1]
                mother = females[i-1]
                noisy_father_phenotype = father_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]), size=father_phenotype.shape)
                noisy_mother_phenotype = mother_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]), size=mother_phenotype.shape)
                father = father[:, np.argsort(noisy_father_phenotype)[0]]
                mother = mother[:, np.argsort(noisy_mother_phenotype)[0]]
                father_phenotype_sorted = father_phenotype[0][np.argsort(noisy_father_phenotype)]
                mother_phenotype_sorted = mother_phenotype[0][np.argsort(noisy_mother_phenotype)]
                corrs[i-1] = np.corrcoef(father_phenotype_sorted, mother_phenotype_sorted)

                orders = np.random.permutation(range(father.shape[1]))
                son_fams = orders[[i%2==0 for i in range(len(orders))]]
                daughter_fams = orders[[i%2!=0 for i in range(len(orders))]]

                son_fathers = father[:, son_fams]
                son_mothers = mother[:, son_fams]
                daughter_fathers = father[:, daughter_fams]
                daughter_mothers = mother[:, daughter_fams]
                son1 = (son_fathers==2) + (son_fathers==1)*np.random.binomial(1, 0.5, son_fathers.shape) + (son_mothers==2) + (son_mothers==1)*np.random.binomial(1, 0.5, son_mothers.shape)
                son2 = (son_fathers==2) + (son_fathers==1)*np.random.binomial(1, 0.5, son_fathers.shape) + (son_mothers==2) + (son_mothers==1)*np.random.binomial(1, 0.5, son_mothers.shape)
                daughter1 = (daughter_fathers==2) + (daughter_fathers==1)*np.random.binomial(1, 0.5, daughter_fathers.shape) + (daughter_mothers==2) + (daughter_mothers==1)*np.random.binomial(1, 0.5, daughter_mothers.shape)
                daughter2 = (daughter_fathers==2) + (daughter_fathers==1)*np.random.binomial(1, 0.5, daughter_fathers.shape) + (daughter_mothers==2) + (daughter_mothers==1)*np.random.binomial(1, 0.5, daughter_mothers.shape)

                males[i] = np.hstack((son1, son2))
                females[i] = np.hstack((daughter1, daughter2))

                male_phenotypes[i] = a@males[i] + b@(np.hstack((son_fathers, son_fathers)) + np.hstack((son_mothers, son_mothers))) + np.random.normal(0, sqrt(ve), (1, pop_size//2))
                female_phenotypes[i] = a@females[i] + b@(np.hstack((daughter_fathers, daughter_fathers)) + np.hstack((daughter_mothers, daughter_mothers))) + np.random.normal(0, sqrt(ve), (1, pop_size//2))
                vy[i] = np.var(np.hstack((male_phenotypes[i][0], female_phenotypes[i][0])))
            all_corrs[(vb, ab_cor, p)] = corrs


import _pickle as pickle
with open("corrs.pickle", "wb") as f:
    pickle.dump({"length": length,
                    "vg0": vg0,
                    "c": 1-np.eye(number_of_the_genes),
                    "p": p,
                    "va": va,
                    "ve": ve,
                    "data_keys": "vb, ab_cor, p",
                    "data_vals": "corrs",
                    "number_of_the_genes": number_of_the_genes,
                    "data":all_corrs},
                f)