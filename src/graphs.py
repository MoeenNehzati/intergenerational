from utils import *
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
np.random.seed(11523)
number_of_the_genes = 1000
length = 30
vg0 = np.eye(number_of_the_genes)/number_of_the_genes
c = 1 - np.eye(number_of_the_genes)
p = 0.5
ve = 1
a = []
b = []
def simulate_and_plot(length, vg0, c, p, va, vb, cor_ab, ve, number_of_the_genes, plot = True):
    ab_cov = cor_ab * np.sqrt(va*vb)
    ab_cov = [[va, ab_cov], [ab_cov, vb]]
    ab = multivariate_normal.rvs(cov = ab_cov, size=number_of_the_genes)
    a = ab[:,0].reshape((1, -1))
    b = ab[:,1].reshape((1, -1))
    vg, vy, m, h2, sib_cov = simulate_generations(length, vg0, c, a, b, p, ve)
    if plot:
        title = f'Vy with p={p}, Ve={ve}, number_of_the_genes={number_of_the_genes},\n Vg0=I{number_of_the_genes}^-1, A,B~N(va={va}, vb={vb}, corr={cor_ab})'
        fig = sns.lineplot("generation", "Vy", data=pd.DataFrame({"Vy":vy, "generation":range(length)}))
        fig.set_title(title)
        fig.get_figure().savefig("../out/figs/"+title.replace(" ", "").replace("\n", "")+".png")
        plt.clf()
        title = f'h2 with p={p}, Ve={ve}, number_of_the_genes={number_of_the_genes},\n Vg0=I{number_of_the_genes}^-1, A,B~N(va={va}, vb={vb}, corr={cor_ab})'
        fig = sns.lineplot("generation", "h2", data=pd.DataFrame({"h2":[(a@vg[i]@a.T).item()/vy[i] for i in range(length)], "generation":range(length)}))
        fig.set_title(title)
        fig.get_figure().savefig("../out/figs/"+title.replace(" ", "").replace("\n", "")+".png")
        plt.show()
        plt.clf()
    return vg, vy, m, h2, sib_cov

# vgsi, vysi, msi, hi2 = simulate_and_plot(length, vg0, c, p, va, 0, 0, ve, number_of_the_genes, False)        
# plt.plot(range(length), [v/hi2[0] for v in hi2], label = f"vb={0}, cor_ab={0}", linewidth = 2)
# for vb in [0.25, 0.5, 1]:    
#     for cor_ab in [0, 0.5, 1]:
#         print(vb, cor_ab)
#         vgsi, vysi, msi, hi2 = simulate_and_plot(length, vg0, c, p, va, vb, cor_ab, ve, number_of_the_genes, False)        
#         plt.plot(range(length), [v/hi2[0] for v in hi2], label = f"vb={vb}, cor_ab={cor_ab}", linewidth = 2)

# plt.legend()
# plt.xlabel('generation')
# plt.ylabel('normalized h2')
# title = f'h2 With p={p}, Ve={ve}, number_of_the_genes={number_of_the_genes},\n Vg0=I{number_of_the_genes}^-1'
# plt.title(title)
# plt.savefig(title.replace(" ", "").replace("\n", "") + ".png")
# plt.clf()
                                      #       (length, vg0, c, p, va, vb, cor_ab, ve, number_of_the_genes, plot = True)
# vgsi, vysi, msi, hi2, sibs = simulate_and_plot(length, vg0, c, p, 1, 0.25, 1, 1, number_of_the_genes, False)        

# vgsi, vysi, msi, hi2 = simulate_and_plot(length, vg0, c, p, va, 0, 0, ve, number_of_the_genes, False)        
# plt.plot(range(length), [v/vysi[0] for v in vysi], label = f"vb={0}, cor_ab={0}", linewidth = 2)
# for vb in [0.25, 0.5, 1]:    
#     for cor_ab in [0, 0.5, 1]:
#         print(vb, cor_ab)
#         vgsi, vysi, msi, hi2 = simulate_and_plot(length, vg0, c, p, va, vb, cor_ab, ve, number_of_the_genes, False)        
#         plt.plot(range(length), [v/vysi[0] for v in vysi], label = f"vb={vb}, cor_ab={cor_ab}", linewidth = 2)

# plt.legend()
# plt.xlabel('generation')
# plt.ylabel('normalized Vy')
# title = f'Vy With p={p}, Ve={ve}, number_of_the_genes={number_of_the_genes},\n Vg0=I{number_of_the_genes}^-1'
# plt.title(title)
# plt.savefig(title.replace(" ", "").replace("\n", "") + ".png")
# plt.clf()



# data = {}
# for vb in [0, 0.25, 0.5, 0.75, 1]:
#     for cor_ab in [0, 0.25, 0.5, 0.75, 1]:
#         for p in [0, 0.25, 0.5, 0.75, 1]:
#             print(vb, cor_ab)
#             vgsi, vysi, msi, h2, sib_corri = simulate_and_plot(length, vg0, c, p, va, vb, cor_ab, ve, number_of_the_genes, False)        
#             data[(vb, cor_ab, p)] = vgsi, vysi, msi, hi2, sib_corri

# import _pickle as pickle
# with open("vbcor.pickle", "wb") as f:
#     pickle.dump({"length": length,
#                     "vg0": vg0,
#                     "c": c,
#                     "p": p,
#                     "va": va,
#                     "vb": vb,
#                     "cor_ab": cor_ab,
#                     "ve": ve,
#                     "data_keys": "vb, cor_ab, p",
#                     "data_vals": "vgsi, vysi, msi, h2, sib_corri",
#                     "number_of_the_genes": number_of_the_genes,
#                     "data":data},
#                 f)



# data = {}
# vgsi, vysi, msi, hi2, sib_corri = simulate_and_plot(length, vg0, c, p, va, 0, 0, ve, number_of_the_genes, False)
# data[(0,0)] = vgsi, vysi, msi, hi2, sib_corri
# for vb in [0.25, 0.5, 1]:    
#     for cor_ab in [0, 0.5, 1]:
#         print(vb, cor_ab)
#         vgsi, vysi, msi, h2, sib_corri = simulate_and_plot(length, vg0, c, p, va, vb, cor_ab, ve, number_of_the_genes, False)        
#         data[(vb, cor_ab)] = vgsi, vysi, msi, hi2, sib_corri

# import _pickle as pickle
# with open("vbcor.pickle", "wb") as f:
#     pickle.dump({"length": length,
#                     "vg0": vg0,
#                     "c": c,
#                     "p": p,
#                     "va": va,
#                     "vb": vb,
#                     "cor_ab": cor_ab,
#                     "ve": ve,
#                     "data_keys": "vb, cor_ab",
#                     "data_vals": "vgsi, vysi, msi, h2, sib_corri",
#                     "number_of_the_genes": number_of_the_genes,
#                     "data":data},
#                 f)

# plt.legend()
# plt.xlabel('generation')
# plt.ylabel('normalized Vy')
# title = f'sib_corr With p={p}, Ve={ve}, number_of_the_genes={number_of_the_genes},\n Vg0=I{number_of_the_genes}^-1'
# plt.title(title)
# plt.savefig(title.replace(" ", "").replace("\n", "") + ".png")
# plt.clf()
