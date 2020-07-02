from utils import *
from pop_sim import simulate
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import _pickle as pickle
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
# np.random.seed(11523)
def f(number_of_the_genes, causal_genes, length, vg0, c, p, ve, a, b, pop_size):
    print("AAAA")
    # number_of_the_genes = 1000
    # causal_genes = number_of_the_genes
    # length = 30
    # vg0 = np.eye(number_of_the_genes)/number_of_the_genes
    # mu = 0
    # c = 1 - np.eye(number_of_the_genes)
    # p = 0.5
    # ve = 1
    # a = np.random.normal(size=(1, number_of_the_genes))
    # pop_size = 20#400000
    
    real = []
    formula = []
    result = simulate(pop_size, length, number_of_the_genes, causal_genes, a, b, vg0, p, ve, 30, True)
    for i in range(length-1):
        female_father_ranks = result["female_father_ranks"]
        female_mother_ranks = result["female_mother_ranks"]
        male_father_ranks = result["male_father_ranks"]
        male_mother_ranks = result["male_mother_ranks"]
        male_phenotypes = result["male_phenotypes"]
        female_phenotypes = result["female_phenotypes"]
        males = result["males"]
        females = result["females"]
        fathers_ranks = np.hstack((female_father_ranks[i+1], male_father_ranks[i+1]))
        mothers_ranks = np.hstack((female_mother_ranks[i+1], male_mother_ranks[i+1]))
        mg = males[i][:, fathers_ranks]
        my = male_phenotypes[i][:, fathers_ranks]
        fg = females[i][:, mothers_ranks]
        fy = female_phenotypes[i][:, mothers_ranks]    
        effect_v = np.var(np.hstack((a@mg, a@fg)))#.item()
        real_cor = np.cov(a@mg, a@fg)[0,1]/effect_v
        t = (np.cov(a@mg, my)[0,1] * np.cov(my, fy)[0,1] * np.cov(fy, a@fg)[0,1])/(np.var(np.hstack((my, fy)))**2)
        formula_cor = t/effect_v
        real.append(real_cor)
        formula.append(formula_cor)
    return {"real":real,
            "formula":formula}


runs = 1000#200
processes = 15
with Pool(processes=processes) as pool:         # start 4 worker processes
    number_of_the_genes = 1000
    causal_genes = number_of_the_genes
    length = 30
    vg0 = np.eye(number_of_the_genes)/2
    mu = 0
    c = 1 - np.eye(number_of_the_genes)
    p = 0.5
    ve = 1
    pop_size = 50000
    a = np.random.normal(size=(1, number_of_the_genes))
    b = [[0]*number_of_the_genes]
    args = [(number_of_the_genes, causal_genes, length, vg0, c, p, ve, a, b, pop_size) for i in range(runs)]
    results = pool.starmap(f, args)
    real = [0] * 29
    formula = [0] * 29
    for result in results:
        reals = result["real"]
        formulas = result["formula"]
    reals = [i/len(results) for i in reals]
    formulas = [i/len(results) for i in formulas]
    plt.plot(range(29), result["real"], label=f"average Cov(ag,ag*)")
    plt.plot(range(29), result["formula"], label=f"average (Cov(a@g,y)*Cov(y,y*)*Cov(y*,a@g*))/Var(y)^2")
    plt.xlabel("time")
    plt.title(f"Average of genetic covariance and its estimate\n over {len(results)} runs")
    plt.legend()
    plt.savefig("gene_corr_break")
    results = {"values":results,
              "params":{"number_of_the_genes": number_of_the_genes,
                        "causal_genes": causal_genes,
                        "length": length,
                        "vg0": vg0,
                        "mu": mu,
                        "c": c,
                        "p": p,
                        "ve": ve,
                        "a": a,
                        "pop_size": pop_size,
                        },
                "description": "Real: average Cov(ag,ag*)\n average (Cov(a@g,y)*Cov(y,y*)*Cov(y*,a@g*))/Var(y)^2",
              }
    with open("gene_corr_break.pickle", "wb") as f:
        pickle.dump(result, f)


# with Pool(processes=processes) as pool:         # start 4 worker processes
#     number_of_the_genes = 1000
#     causal_genes = number_of_the_genes
#     length = 30
#     vg0 = np.eye(number_of_the_genes)/number_of_the_genes
#     mu = 0
#     c = 1 - np.eye(number_of_the_genes)
#     p = 0.5
#     ve = 1
#     a = np.random.normal(size=(1, number_of_the_genes))
#     pop_size = 200000
#     args = [(number_of_the_genes, causal_genes, length, vg0, mu, c, p, ve, a, pop_size) for i in range(runs)]
#     results = pool.starmap(f, args)
#     real = [0] * 29
#     formula = [0] * 29
#     for result in results:
#         reals = result["real"]
#         formulas = result["formula"]
#     reals = [i/len(results) for i in reals]
#     formulas = [i/len(results) for i in formulas]
#     plt.plot(range(29), result["real"], label=f"average Cov(ag,ag*)")
#     plt.plot(range(29), result["formula"], label=f"average (Cov(a@g,y)*Cov(y,y*)*Cov(y*,a@g*))/Var(y)^2")
#     plt.xlabel("time")
#     plt.title(f"Average of genetic covariance and its estimate\n over {len(results)} runs")
#     plt.legend()
#     plt.savefig("gene_corr_break")
#     results = {"values":results,
#               "params":{"number_of_the_genes": number_of_the_genes,
#                         "causal_genes": causal_genes,
#                         "length": length,
#                         "vg0": vg0,
#                         "mu": mu,
#                         "c": c,
#                         "p": p,
#                         "ve": ve,
#                         "a": a,
#                         "pop_size": pop_size,
#                         },
#                 "description": "Real: average Cov(ag,ag*)\n average (Cov(a@g,y)*Cov(y,y*)*Cov(y*,a@g*))/Var(y)^2",
#               }
#     with open("gene_corr_break.pickle", "wb") as f:
#         pickle.dump(result, f)

