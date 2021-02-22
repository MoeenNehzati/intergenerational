from utils import *
from pop_sim import simulate
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import _pickle as pickle
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
import argparse
# np.random.seed(11523)



def computations(i, data, statistics):
    if i>0:        
        females = data["females"]
        males = data["males"]
        male_father_ranks = data["male_father_ranks"]
        female_father_ranks = data["female_father_ranks"]
        male_mother_ranks = data["male_mother_ranks"]
        female_mother_ranks = data["female_mother_ranks"]
        male_phenotypes = data["male_phenotypes"]
        female_phenotypes = data["female_phenotypes"]
        a = data["a"]
        # print(np.hstack((male_father_ranks, female_father_ranks)).dtype)
        father_ranks = np.hstack((male_father_ranks[i], female_father_ranks[i]))
        mother_ranks = np.hstack((male_mother_ranks[i], female_mother_ranks[i]))
        mg = males[i-1][:, father_ranks]
        fg = females[i-1][:, mother_ranks]
        my = male_phenotypes[i-1][0, father_ranks]
        fy = female_phenotypes[i-1][0, mother_ranks]    
        print(np.corrcoef(my, fy))
        effect_v = np.var(np.hstack((a@mg, a@fg)))#.item()
        real_cor = np.cov(a@mg, a@fg)[0,1]/effect_v
        # Cov(ag,ag*) = Cov(E(ag|y), E(ag*|y*)) = Cov( (Cov(ag,y)/Vy)y,  (Cov(ag*,y*)/Vy)y*) = (Cov(ag,y)*Cov(y,y*)*Cov(y*ag*))/var(y)^2
        t = (np.cov(a@mg, my)[0,1] * np.cov(my, fy)[0,1] * np.cov(fy, a@fg)[0,1])/(np.var(np.hstack((my, fy)))**2)
        formula_cor = t/effect_v
        print(effect_v)
        statistics["real_cor"][i-1] = real_cor
        statistics["formula_cor"][i-1] = formula_cor


def f(number_of_the_genes, causal_genes, length, vg0, c, p, ve, a, b, pop_size):
    print("AAAA")    
    real = []
    formula = []
    statistics = {"formula_cor":[None]*length, "real_cor":[None]*length}
    result = simulate(pop_size, length, number_of_the_genes, causal_genes, a, b, vg0, p, ve, 2, computations, statistics, True)
    #TODO set input
    return {"real":statistics["real_cor"],
            "formula":statistics["formula_cor"]}


runs = 15#200
processes = 15
parser = argparse.ArgumentParser()
parser.add_argument('-r', type=int, help='Number of the runs')
parser.add_argument('-p', type=int, help='Number of the processes', default = 15)
parser.add_argument('-pop', type=int, help='population_size')
parser.add_argument('-snps', type=int, help='number of the snps')
parser.add_argument('-maf', type=float, help='min allele frequency')
parser.add_argument('-bvar', type=float, help='variance of indirect effects')
parser.add_argument('-ab_corr', type=float, help='correlation between direct and indirect effect')
parser.add_argument('-out_prefix', type=str, help='defaults to sim', default="sim_")
args=parser.parse_args()
runs = args.r
processes = args.p
pop_size = args.pop
number_of_the_genes = args.snps
maf = args.maf
bvar = args.bvar
ab_corr = args.ab_corr
out_prefix = args.out_prefix
with Pool(processes=processes) as pool:         # start 4 worker processes
    no_dot_maf = str(maf).replace(".","-")
    no_dot_bvar = str(bvar).replace(".","-")
    no_dot_ab_corr = str(ab_corr).replace(".","-")
    param_desc = f"pop{pop_size}_snps{number_of_the_genes}_maf{no_dot_maf}_runs{runs}_bvar{no_dot_bvar}_abcorr{no_dot_ab_corr}"
    causal_genes = number_of_the_genes
    length = 30
    vg0 = np.eye(number_of_the_genes)*maf*(1-maf)*2
    mu = 0
    c = 1 - np.eye(number_of_the_genes)
    p = 0.5
    ve = 1
    ab_cov = ab_corr*bvar
    ab_cov_matrix = [[1, ab_cov],[ab_cov, bvar]]
    ab = multivariate_normal.rvs(cov = ab_cov_matrix, size=number_of_the_genes)
    a = ab[:,0].reshape((1, -1))
    b = ab[:,1].reshape((1, -1))
    # a = np.random.normal(size=(1, number_of_the_genes)) #scale=np.sqrt(1/number_of_the_genes/(maf*(3*maf+1))), 
    sample_g = np.random.binomial(2, maf, (number_of_the_genes, pop_size//2))
    result_var = np.var(a@sample_g)
    a = a/np.sqrt(result_var)
    b = b/np.sqrt(result_var)
    args = [(number_of_the_genes, causal_genes, length, vg0, c, p, ve, a, b, pop_size) for i in range(runs)]
    results = pool.starmap(f, args)
    reals = np.mean([result["real"][:-1] for result in results], axis=0)
    formulas = np.mean([result["formula"][:-1] for result in results], axis=0)
    result = {
        "reals":reals,
        "formulas":formulas,
    }
    plt.plot(range(29), reals, label=f"average Corr(ag,ag*)")
    plt.plot(range(29), formulas, label=f"average (Cov(a@g,y)*Cov(y,y*)*Cov(y*,a@g*))/Var(y)^2")
    plt.xlabel("time")
    plt.title(f"Average of genetic covariance and its estimate\n"+param_desc)
    plt.legend()
    print(out_prefix+param_desc+".png")
    plt.savefig(out_prefix+param_desc)
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
                        "b": b,
                        "pop_size": pop_size,
                        },
                "description": "Real: average Corr(ag,ag*)\n average (Cov(a@g,y)*Cov(y,y*)*Cov(y*,a@g*))/Var(y)^2",
              }
    with open(out_prefix+param_desc+".pickle", "wb") as f:
        pickle.dump(results, f)


