from utils import *
from pop_sim import simulate
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import _pickle as pickle
from sklearn.linear_model import LinearRegression
np.random.seed(11523)
number_of_the_genes = 1000
causal_genes = number_of_the_genes
length = 30
vg0 = np.eye(number_of_the_genes)/number_of_the_genes
mu = 0
c = 1 - np.eye(number_of_the_genes)
p = 0.5
ve = 1
a = np.random.normal(size=(1, number_of_the_genes))
pop_size = 10000
for pop_size in [10000, 1000000, 10000000]:
    result = simulate(pop_size, length, number_of_the_genes, causal_genes, mu, a, vg0, p, ve, 8, True)
    male_paternal_relations = result["male_father_ranks"]
    male_maternal_relations = result["male_mother_ranks"]
    female_paternal_relations = result["female_father_ranks"]
    female_maternal_relations = result["female_mother_ranks"]
    male_phenotypes = result["male_phenotypes"]
    female_phenotypes = result["female_phenotypes"]
    #source is -1
    paired = zip(np.hstack((male_paternal_relations[-1], female_paternal_relations[-1])), 
    np.hstack((male_maternal_relations[-1], female_maternal_relations[-1])))
    male_to_female_spouse_index = {i:j for i,j in paired}
    male_father_index = male_paternal_relations[-2]
    male_grandfather_index = male_paternal_relations[-3][male_father_index]
    male_greatgrandfather_index = male_paternal_relations[-4][male_grandfather_index]

    female = female_phenotypes[-2][:,[male_to_female_spouse_index[i] for i in range(pop_size//2)]]
    male = male_phenotypes[-2]
    male_fathers = male_phenotypes[-3][:,male_father_index]
    male_grandfathers = male_phenotypes[-4][:,male_grandfather_index]
    male_greatgrandfathers = male_phenotypes[-5][:,male_greatgrandfather_index]

    X = np.hstack((male.T, male_fathers.T, male_grandfathers.T, male_greatgrandfathers.T))
    Y = female.T
    reg = LinearRegression().fit(X, Y)
    print(pop_size)
    print(reg.coef_)
    print(reg.intercept_)
