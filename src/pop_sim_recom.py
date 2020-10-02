import numpy as np
from tqdm import tqdm
from numpy import sqrt
from time import time
from time import time
from bgen_reader import read_bgen
import pandas as pd
import os
import _pickle as pickle

def prepare_data(chr_range, bgen_address = "/disk/genetics4/ukb/alextisyoung/haplotypes/sim_genotypes/QCed/chr_~", start=0, end=None):
    from_chr = min(chr_range)
    to_chr = max(chr_range)
    #TODO fix address
    loaded_address = f"/disk/genetics4/ukb/alextisyoung/haplotypes/sim_genotypes/QCed/cache/loaded_data_from{from_chr}_to{to_chr}_start{start}_end{end}.pickle"
    if os.path.isfile(loaded_address):
        print("cache exists")
        with open(loaded_address, "rb") as f:
           phased_gts, map_list, chrom_lengths = pickle.load(f)
           return phased_gts, map_list, chrom_lengths
    
    chr_starts = np.array(chr_range)
    chrom_lengths =[]
    whole_genotype = []
    poss = []
    for chrom in chr_range:
        print("reading chromosome", chrom)
        bgen = read_bgen(bgen_address.replace("~", str(chrom))+".bgen", verbose=False)
        bgen_bim = bgen["variants"].compute().rename(columns={"chrom":"Chr", "pos":"coordinate"})
        variants_number = bgen_bim.shape[0]
        this_end = end
        if this_end is None:
            this_end = variants_number
        poss.append(bgen_bim["coordinate"].values[start:this_end])
        gts_ids = bgen["samples"].values
        pop_size = len(gts_ids)
        chromosome_genotype = []        
        chrom_lengths.append(this_end - start)
        counter = 1
        for genotype in tqdm(bgen["genotype"][start:this_end]):
            counter+=1
            computed = genotype.compute()
            probs = computed["probs"]
            genomes = np.array([(None, None)]*pop_size)
            genomes[probs[:,0] > 0.99, 0] = 1.
            genomes[probs[:,1] > 0.99, 0] = 0.
            genomes[probs[:,2] > 0.99, 1] = 1.
            genomes[probs[:,3] > 0.99, 1] = 0.
            chromosome_genotype.append(genomes)
        whole_genotype = whole_genotype + chromosome_genotype
    phased_gts = np.array(whole_genotype, dtype=float)#.transpose(1,0,2)
    print("phased_gts.shape", phased_gts.shape)
    if phased_gts.shape[1]%4 != 0:
        phased_gts = phased_gts[:, :-(phased_gts.shape[1]%4), :]
    pop_size = phased_gts.shape[1]
    map_file = pd.read_csv("../genetic_map_hg19_withX.txt", sep=" ")
    map_file.columns = ["chr", "position", "cmmb", "cm"]
    map_for_chrs = []
    for chr in range(1, 23):
        this_chr_map_file = map_file[map_file["chr"] == chr]
        pos = this_chr_map_file["position"].values.reshape((-1, 1))
        cm = this_chr_map_file["cm"].values.reshape((-1, 1))
        zipped = np.hstack((pos,cm))
        map_for_chrs.append(zipped)
    map_list = get_cms(chr_range, poss, map_for_chrs)
    map_list = [[i if i is not None else 0 for i in arr] for arr in map_list]
    with open(loaded_address, "wb") as f:
        pickle.dump((phased_gts, map_list, chrom_lengths), f)
    return phased_gts, map_list, chrom_lengths
    #TODO handle these nans

def get_cms(chrs, poss, maps):
    cm_for_chrs = []
    for i in range(len(chrs)):        
        this_chr_cms = []
        chr = chrs[i]
        pos = poss[i]
        map = maps[chr-1]
        for p in pos:
            end_index = np.argmax(map[:, 0]>=p)
            if map[end_index, 0] == p:
                    result = map[end_index, 1]
            elif end_index > 0:
                    p_end = map[end_index, 0]
                    p_start = map[end_index-1, 0]
                    c_end = map[end_index, 1]
                    c_start = map[end_index-1, 1]
                    result = c_start+(p-p_start)*(c_end-c_start)/(p_end-p_start)
            else:
                result = None
            this_chr_cms.append(result)
        cm_for_chrs.append(this_chr_cms)
    return cm_for_chrs
#TODO handle None

def mate(fathers, mothers, map_list, chrom_lengths):
    recombed_fathers = recombination(fathers, map_list, chrom_lengths)
    recombed_mothers = recombination(mothers, map_list, chrom_lengths)
    offsprings = np.stack((recombed_fathers, recombed_mothers), axis=2)
    return offsprings

def recombination(population, map_list, chrom_lengths):
    #need 22 chrom length
    snps, individuals = population.shape[:2]
    recombs = np.zeros((snps, individuals))
    counters = [0 for i in range(individuals)]
    for chr in range(len(chrom_lengths)):
        length = chrom_lengths[chr]
        length_in_centimorgans = np.max(map_list[chr])
        length_in_morgans = int(length_in_centimorgans/100)+1
        sample_size = int(2*length_in_morgans)
        intervals = np.random.exponential(1/length_in_morgans, (individuals, sample_size))
        intervals = np.hstack((intervals, np.array([[1.1] for i in range(individuals)])))
        points = np.cumsum(intervals, axis=1)
        points = points*length_in_centimorgans
        # points = points.astype(int)        
        for i in range(individuals):        
            counter = counters[i]
            splits = int(np.random.random()>0.5)
            for k in range(length):
                while map_list[chr][k]>points[i, splits]:
                    splits += 1
                recombs[counter,i] = population[counter, i, splits%2]
                counter+=1
            counters[i] = counter
    return recombs



def recombination_shadow(population, map_list, chrom_lengths):
    snps, individuals = population.shape[:2]
    length_in_centimorgans = np.max(map_list)
    length_in_morgans = int(length_in_centimorgans/100)+1
    sample_size = int(4*length_in_morgans)
    intervals = np.random.exponential(1/length_in_morgans, (individuals, sample_size))
    intervals = np.hstack((intervals, np.array([[1] for i in range(individuals)])))
    points = np.cumsum(intervals, axis=1)
    points = points*length_in_centimorgans
    points = points.astype(int)
    recombs = np.zeros((snps, individuals))
    for i in range(individuals):        
        counter = 0
        splits = 0
        for length in chrom_lengths:
            mode = np.random.random()>0.5
            for k in range(length):
                while map_list[counter]>points[i, splits]:
                    splits += 1
                recombs[counter,i] = population[counter, i, (mode+splits)%2]
                counter+=1
    return recombs

def simulate(length,
            causal_genes,
            phased_gts,
            map_list,
            chrom_lengths,            
            a,
            b,
            p,
            ve,
            latest_mem,            
            statistics_function = None,
            statistics = {},
            force_p=True,            
            ):

    pop_size = phased_gts.shape[1]
    number_of_genes = phased_gts.shape[0]

    if latest_mem is None:
        latest_mem = length
    males = [None]*length
    females = [None]*length
    
    female_father_ranks = np.zeros((length, pop_size//2)).astype(int)
    female_mother_ranks = np.zeros((length, pop_size//2)).astype(int)
    male_father_ranks = np.zeros((length, pop_size//2)).astype(int)
    male_mother_ranks = np.zeros((length, pop_size//2)).astype(int)
    male_phenotypes = np.array([[None]*(pop_size//2)]*length, dtype=float)
    female_phenotypes = np.array([[None]*(pop_size//2)]*length, dtype=float)
    male_direct = np.array([[None]*(pop_size//2)]*length, dtype=float)
    male_indirect = np.array([[None]*(pop_size//2)]*length, dtype=float)
    female_direct = np.array([[None]*(pop_size//2)]*length, dtype=float)
    female_indirect = np.array([[None]*(pop_size//2)]*length, dtype=float)
    cov_parental_direct = np.array([None]*length, dtype=float)
    cov_parental_indirect = np.array([None]*length, dtype=float)
    cov_parental_direct_indirect = np.array([None]*length, dtype=float)
    vy = np.array([None]*length, dtype=float)
    v_direct = np.array([None]*length, dtype=float)
    v_indirect = np.array([None]*length, dtype=float)
    cov_direct_indirect = np.array([None]*length, dtype=float)
    heritablity = np.array([None]*length, dtype=float)
    corrs = np.array([None]*length, dtype=float)
    corrs_original = np.array([None]*length, dtype=float)
    corrs_inbreeding = np.array([None]*length, dtype=float)
    corrs_causal = np.array([None]*length, dtype=float)
    deltas = np.array([None]*length, dtype=float)
    #lets say males are odds and females are even
    males[0] = phased_gts[:, [i%2==0 for i in range(pop_size)], :].astype(float)
    females[0] = phased_gts[:, [i%2==1 for i in range(pop_size)], :].astype(float)
    male_phenotypes[0,:] = a@np.sum(males[0], axis=2) + np.random.normal(0, sqrt(ve), (1, pop_size//2))
    female_phenotypes[0,:] = a@np.sum(females[0], axis=2) + np.random.normal(0, sqrt(ve), (1, pop_size//2))
    vy[0] = np.var(np.hstack((male_phenotypes[0, :], female_phenotypes[0, :])))
    data = {
            "males":males,
            "females":females,
            "female_father_ranks":female_father_ranks,
            "female_mother_ranks":female_mother_ranks,
            "male_father_ranks":male_father_ranks,
            "male_mother_ranks":male_mother_ranks,
            "male_phenotypes":male_phenotypes,
            "female_phenotypes":female_phenotypes,
            "a":a,
            "b":b,
        }
    for i in tqdm(range(1, length)):
        if i >= latest_mem:
            males[i-latest_mem] = None
            females[i-latest_mem] = None

        delta = 0.9
        dif = 1
        while dif>=0:
            delta += 0.1
            father_phenotype = male_phenotypes[i-1, :]
            mother_phenotype = female_phenotypes[i-1, :]
            father = males[i-1]
            mother = females[i-1]
            noisy_father_phenotype = father_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=father_phenotype.shape)
            noisy_mother_phenotype = mother_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=mother_phenotype.shape)
            father = father[:, np.argsort(noisy_father_phenotype),:]
            mother = mother[:, np.argsort(noisy_mother_phenotype),:]
            father_phenotype_sorted = father_phenotype[np.argsort(noisy_father_phenotype)]
            mother_phenotype_sorted = mother_phenotype[np.argsort(noisy_mother_phenotype)]
            temp_cor = np.corrcoef(father_phenotype_sorted, mother_phenotype_sorted)
            dif = temp_cor[0,1]-p
            if delta == 1:
                corrs_original[i] = temp_cor[0,1]
            if not force_p:
                break
        while dif<0 and force_p:
            delta -= 0.01
            father_phenotype = male_phenotypes[i-1, :]
            mother_phenotype = female_phenotypes[i-1, :]
            father = males[i-1]
            mother = females[i-1]
            noisy_father_phenotype = father_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=father_phenotype.shape)
            noisy_mother_phenotype = mother_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=mother_phenotype.shape)
            father = father[:, np.argsort(noisy_father_phenotype), :]
            mother = mother[:, np.argsort(noisy_mother_phenotype), :]
            father_phenotype_sorted = father_phenotype[np.argsort(noisy_father_phenotype)]
            mother_phenotype_sorted = mother_phenotype[np.argsort(noisy_mother_phenotype)]
            temp_cor = np.corrcoef(father_phenotype_sorted, mother_phenotype_sorted)
            dif = temp_cor[0,1]-p
        sorted_to_initial_ranks_male = np.argsort(noisy_father_phenotype)
        sorted_to_initial_ranks_female = np.argsort(noisy_mother_phenotype)
        deltas[i] = delta
        corrs[i-1] = temp_cor[0,1]
        corrs_inbreeding[i-1] = np.corrcoef(np.sum(father[causal_genes:,:,:], axis=2).reshape((1, -1)), np.sum(mother[causal_genes:,:,:], axis=2).reshape((1, -1)))[0,1]
        corrs_causal[i-1] = np.corrcoef(np.sum(father[:causal_genes,:,:], axis=2).reshape((1, -1)), np.sum(mother[:causal_genes,:,:], axis=2).reshape((1, -1)))[0,1]
    
        orders = np.random.permutation(range(father.shape[1]))
        son_fams = orders[[i%2==0 for i in range(len(orders))]]
        daughter_fams = orders[[i%2!=0 for i in range(len(orders))]]
        son_fathers = father[:, son_fams, :]
        son_mothers = mother[:, son_fams, :]
        daughter_fathers = father[:, daughter_fams, :]
        daughter_mothers = mother[:, daughter_fams, :]
        son1 = mate(son_fathers, son_mothers, map_list, chrom_lengths)
        son2 = mate(son_fathers, son_mothers, map_list, chrom_lengths)
        daughter1 = mate(daughter_fathers, daughter_mothers, map_list, chrom_lengths)
        daughter2 = mate(daughter_fathers, daughter_mothers, map_list, chrom_lengths)
        males[i] = np.hstack((son1, son2))
        females[i] = np.hstack((daughter1, daughter2))

        male_parent_sorted_ranks = np.hstack((son_fams, son_fams))
        male_father_ranks[i] = np.array([sorted_to_initial_ranks_male[male_parent_sorted_ranks[j]] for j in range(pop_size//2)])
        male_mother_ranks[i] = np.array([sorted_to_initial_ranks_female[male_parent_sorted_ranks[j]] for j in range(pop_size//2)])

        female_parent_sorted_ranks = np.hstack((daughter_fams, daughter_fams))
        female_father_ranks[i] = np.array([sorted_to_initial_ranks_male[female_parent_sorted_ranks[j]] for j in range(pop_size//2)])
        female_mother_ranks[i] = np.array([sorted_to_initial_ranks_female[female_parent_sorted_ranks[j]] for j in range(pop_size//2)])
        
        if statistics_function is not None:
            statistics_function(i, data, statistics)
        male_direct[i, :] = a@np.sum(males[i],2)
        male_indirect[i, :] = b@(np.sum(np.hstack((son_fathers, son_fathers)), 2) + np.sum(np.hstack((son_mothers, son_mothers)), 2))
        female_direct[i, :] = a@np.sum(females[i],2)
        female_indirect[i, :] = b@(np.sum(np.hstack((daughter_fathers, daughter_fathers)), 2) + np.sum(np.hstack((daughter_mothers,  daughter_mothers)), 2))
        male_phenotypes[i, :] = male_direct[i, :] + male_indirect[i, :] + np.random. normal(0, sqrt(ve), (1, pop_size//2))
        female_phenotypes[i, :] = female_direct[i, :] + female_indirect[i, :] + np.random.normal(0, sqrt(ve), (1, pop_size//2))

        vy[i] = np.var(np.hstack((male_phenotypes[i, :], female_phenotypes[i, :])))
        direct_indirect_cov_mat = np.cov(np.hstack((male_direct[i, :], female_direct[i, :])).flatten(), np.hstack((male_indirect[i, :], female_indirect[i, :])).flatten())        
        v_direct[i] = direct_indirect_cov_mat[0,0]
        v_indirect[i] = direct_indirect_cov_mat[1,1]
        cov_direct_indirect[i] = direct_indirect_cov_mat[0, 1]
        heritablity[i] = v_direct[i]/vy[i]
        father_direct = np.hstack((male_direct[i-1, female_father_ranks[i]], male_direct[i-1, male_father_ranks[i]]))
        mother_direct = np.hstack((female_direct[i-1, female_mother_ranks[i]], female_direct[i-1, male_mother_ranks[i]]))
        father_indirect = np.hstack((male_indirect[i-1, female_father_ranks[i]], male_indirect[i-1, male_father_ranks[i]]))
        mother_indirect = np.hstack((female_indirect[i-1, female_mother_ranks[i]], female_indirect[i-1, male_mother_ranks[i]]))
        cov_parental_direct[i] = np.cov(father_direct, mother_direct)[0,1]
        cov_parental_indirect[i] = np.cov(father_indirect, mother_direct)[0,1]
        cov_parental_direct_indirect[i] = np.cov(np.hstack((father_direct, mother_direct)), np.hstack((mother_indirect, father_indirect)))[0,1]

    
    return {"vy":vy, 
            "corrs":corrs,
            "corrs_causal":corrs_causal,
            "corrs_inbreeding":corrs_inbreeding,
            "male_phenotypes":male_phenotypes,
            "female_phenotypes":female_phenotypes,
            "male_direct":male_direct,
            "male_indirect":male_indirect,
            "female_direct":female_direct,
            "female_indirect":female_indirect,
            "deltas":deltas,
            "males":np.array([gen.tolist() for gen in males if gen is not None]),
            "females":np.array([gen.tolist() for gen in females if gen is not None]),
            "female_father_ranks": female_father_ranks,
            "female_mother_ranks": female_mother_ranks,
            "male_father_ranks": male_father_ranks,
            "male_mother_ranks": male_mother_ranks,
            "v_direct": v_direct,
            "v_indirect": v_indirect,
            "cov_direct_indirect": cov_direct_indirect,
            "heritablity": heritablity,
            "cov_parental_direct": cov_parental_direct,
            "cov_parental_indirect": cov_parental_indirect,
            "cov_parental_direct_indirect": cov_parental_direct_indirect,
        }