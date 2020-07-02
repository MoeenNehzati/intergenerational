import numpy as np 
from tqdm import tqdm
from numpy import sqrt 

def simulate(pop_size, length, number_of_the_genes, causal_genes, a, b, vg0, p, ve, latest_mem, force_p=True):
    f = 0
    for i in np.roots([2, -2, vg0[0, 0]]):
        if 0 < i <= 0.5:
            f = i
            break
    
    if f == 0:
        raise Exception()
    if latest_mem is None:
        latest_mem = length
    
    males = [None]*length
    females = [None]*length
    female_father_ranks = [None]*length
    female_mother_ranks = [None]*length
    male_father_ranks = [None]*length
    male_mother_ranks = [None]*length

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
        if i >= latest_mem:
            males[i-latest_mem] = None
            females[i-latest_mem] = None
            female_father_ranks[i-latest_mem] = None
            female_mother_ranks[i-latest_mem] = None
            male_father_ranks[i-latest_mem] = None
            male_mother_ranks[i-latest_mem] = None
            male_phenotypes[i-latest_mem] = None
            female_phenotypes[i-latest_mem] = None
            vy[i-latest_mem] = None
            corrs[i-latest_mem] = None
            corrs_original[i-latest_mem] = None
            corrs_inbreeding[i-latest_mem] = None
            corrs_causal[i-latest_mem] = None
            deltas[i-latest_mem] = None

        delta = 0.9
        dif = 1
        while dif>=0:
            delta += 0.1
            father_phenotype = male_phenotypes[i-1]
            mother_phenotype = female_phenotypes[i-1]
            father = males[i-1]
            mother = females[i-1]
            noisy_father_phenotype = father_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=father_phenotype.shape)
            noisy_mother_phenotype = mother_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=mother_phenotype.shape)
            father = father[:, np.argsort(noisy_father_phenotype)[0]]
            mother = mother[:, np.argsort(noisy_mother_phenotype)[0]]
            father_phenotype_sorted = father_phenotype[0][np.argsort(noisy_father_phenotype)]
            mother_phenotype_sorted = mother_phenotype[0][np.argsort(noisy_mother_phenotype)]
            temp_cor = np.corrcoef(father_phenotype_sorted, mother_phenotype_sorted)
            dif = temp_cor[0,1]-p
            if delta == 1:
                corrs_original[i] = temp_cor
            if not force_p:
                break
        while dif<0 and force_p:
            delta -= 0.01
            father_phenotype = male_phenotypes[i-1]
            mother_phenotype = female_phenotypes[i-1]
            father = males[i-1]
            mother = females[i-1]
            noisy_father_phenotype = father_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=father_phenotype.shape)
            noisy_mother_phenotype = mother_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=mother_phenotype.shape)
            father = father[:, np.argsort(noisy_father_phenotype)[0]]
            mother = mother[:, np.argsort(noisy_mother_phenotype)[0]]
            father_phenotype_sorted = father_phenotype[0][np.argsort(noisy_father_phenotype)]
            mother_phenotype_sorted = mother_phenotype[0][np.argsort(noisy_mother_phenotype)]
            temp_cor = np.corrcoef(father_phenotype_sorted, mother_phenotype_sorted)
            dif = temp_cor[0,1]-p
        sorted_to_initial_ranks_male = np.argsort(noisy_father_phenotype)[0]
        sorted_to_initial_ranks_female = np.argsort(noisy_mother_phenotype)[0]
        deltas[i] = delta
        corrs[i-1] = temp_cor
        corrs_inbreeding[i-1] = np.corrcoef(father[causal_genes:,:].reshape((1, -1)), mother[causal_genes:,:].reshape((1, -1)))
        corrs_causal[i-1] = np.corrcoef(father[:causal_genes,:].reshape((1, -1)), mother[:causal_genes,:].reshape((1, -1)))
    
        orders = np.random.permutation(range(father.shape[1]))
        son_fams = orders[[i%2==0 for i in range(len(orders))]]
        daughter_fams = orders[[i%2!=0 for i in range(len(orders))]]
    
        son_fathers = father[:, son_fams]
        son_mothers = mother[:, son_fams]
        daughter_fathers = father[:, daughter_fams]
        daughter_mothers = mother[:, daughter_fams]
        son1 = (son_fathers==2) + (son_fathers==1)*np.random.binomial(1, 0.5, son_fathers.shape) + (son_mothers==2) + (son_mothers==1)*np.  random.binomial(1, 0.5, son_mothers.shape)
        son2 = (son_fathers==2) + (son_fathers==1)*np.random.binomial(1, 0.5, son_fathers.shape) + (son_mothers==2) + (son_mothers==1)*np.  random.binomial(1, 0.5, son_mothers.shape)
        daughter1 = (daughter_fathers==2) + (daughter_fathers==1)*np.random.binomial(1, 0.5, daughter_fathers.shape) +(daughter_mothers==2) + (daughter_mothers==1)*np.random.binomial(1, 0.5, daughter_mothers.shape)
        daughter2 = (daughter_fathers==2) + (daughter_fathers==1)*np.random.binomial(1, 0.5, daughter_fathers.shape) +  (daughter_mothers==2) + (daughter_mothers==1)*np.random.binomial(1, 0.5, daughter_mothers.shape)
        
        males[i] = np.hstack((son1, son2))
        females[i] = np.hstack((daughter1, daughter2))

        male_parent_sorted_ranks = np.hstack((son_fams, son_fams))
        male_father_ranks[i] = np.array([sorted_to_initial_ranks_male[male_parent_sorted_ranks[j]] for j in range(pop_size//2)])
        male_mother_ranks[i] = np.array([sorted_to_initial_ranks_female[male_parent_sorted_ranks[j]] for j in range(pop_size//2)])

        female_parent_sorted_ranks = np.hstack((daughter_fams, daughter_fams))
        female_father_ranks[i] = np.array([sorted_to_initial_ranks_male[female_parent_sorted_ranks[j]] for j in range(pop_size//2)])
        female_mother_ranks[i] = np.array([sorted_to_initial_ranks_female[female_parent_sorted_ranks[j]] for j in range(pop_size//2)])
        
    
        male_phenotypes[i] = a@males[i] + b@(np.hstack((son_fathers, son_fathers)) + np.hstack((son_mothers, son_mothers))) + np.random. normal(0, sqrt(ve), (1, pop_size//2))
        female_phenotypes[i] = a@females[i] + b@(np.hstack((daughter_fathers, daughter_fathers)) + np.hstack((daughter_mothers,  daughter_mothers))) + np.random.normal(0, sqrt(ve), (1, pop_size//2))
        vy[i] = np.var(np.hstack((male_phenotypes[i][0], female_phenotypes[i][0])))
    
    return {"vy":vy, 
            "corrs":corrs,
            "corrs_causal":corrs_causal,
            "corrs_inbreeding":corrs_inbreeding,
            "male_phenotypes":male_phenotypes,
            "female_phenotypes":female_phenotypes,
            "deltas":deltas,
            "males":males,
            "females":females,
            "female_father_ranks": female_father_ranks,
            "female_mother_ranks": female_mother_ranks,
            "male_father_ranks": male_father_ranks,
            "male_mother_ranks": male_mother_ranks,
        }