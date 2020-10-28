import numpy as np
from tqdm import tqdm
from numpy import sqrt
from time import time
from time import time
from bgen_reader import read_bgen
import pandas as pd
import os
import h5py
import logging
from multiprocessing import Pool
from pyplink import PyPlink
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def write_bed(input_address = "outputs/from_chr1_to_chr23_start0_end50_run0_p0-5_ab_corr0-0_vb0-25_length15.hdf5", output_address = None, chunk_size = 100):
    # for x in `ls outputs/*.hdf5`; do python -c "from pop_sim_recom import write_bed;write_bed('$x')"& done
    if output_address is None:
        output_address = input_address[:-5]+"_generation&"
    hf = h5py.File(input_address, "r")
    number_of_snps = hf["males"].shape[1]
    half_pop_size = hf["males"].shape[2]
    iterations = -(-number_of_snps//chunk_size)
    for generation in [0,1]:
        print("for generation", generation)
        with PyPlink(output_address.replace("&", str(generation)), "w") as pedfile:
            for chunk_index in tqdm(range(iterations)):
                male_phased_chunk = np.array(hf["males"][generation,chunk_index*chunk_size:(chunk_index+1)*chunk_size, :, :])
                male_unphased_chunk = np.sum(male_phased_chunk, 2)
                female_phased_chunk = np.array(hf["females"][generation,chunk_index*chunk_size:(chunk_index+1)*chunk_size, :, :])
                female_unphased_chunk = np.sum(female_phased_chunk, 2)                
                unphased_chunk = np.hstack((male_unphased_chunk, female_unphased_chunk))
                for genotypes in unphased_chunk:
                    pedfile.write_genotypes(genotypes)

            # for gender in ["males", "females"]:
            #     print("for", gender)
                # for chunk_index in tqdm(range(iterations)):
                #     phased_chunk = np.array(hf[gender][generation,chunk_index*chunk_size:(chunk_index+1)*chunk_size, :, :])
                #     unphased_chunk = np.sum(phased_chunk, 2)
                #     for genotypes in unphased_chunk:
                #         pedfile.write_genotypes(genotypes)

    columns = [a.decode() for a in hf["bim_columns"][:]]
    values = hf["bim_values"][:].astype(str) 
    original_bim = pd.DataFrame(values, columns = columns)
    splited_alleles = original_bim["allele_ids"].str.split(",", expand = True)
    original_bim["allele1"] = splited_alleles[0]
    original_bim["allele2"] = splited_alleles[1]
    original_bim["base_pair"] = 1
    output_bim = original_bim[["Chr", "rsid", "coordinate", "base_pair", "allele1", "allele2"]]    

    male_phenotypes = hf["male_phenotypes"][-2:, :]
    female_phenotypes = hf["female_phenotypes"][-2:, :]
    female_mother_ranks = hf["female_mother_ranks"][-2:, :]
    female_father_ranks = hf["female_father_ranks"][-2:, :]
    male_mother_ranks = hf["male_mother_ranks"][-2:, :]
    male_father_ranks = hf["male_father_ranks"][-2:, :]

    fam0 = pd.DataFrame([["FID", "IID", "PID", "MID", "SEX", "PHEN"] for i in range(2*half_pop_size)], columns = ["FID", "IID", "PID", "MID", "SEX", "PHEN"])
    fam1 = pd.DataFrame([["FID", "IID", "PID", "MID", "SEX", "PHEN"] for i in range(2*half_pop_size)], columns = ["FID", "IID", "PID", "MID", "SEX", "PHEN"])
    base_iid = 1
    iids = list(range(base_iid, base_iid+2*half_pop_size))
    base_iid += 2*half_pop_size
    fam0["IID"] = iids
    fam0["PID"] = 0
    fam0["MID"] = 0
    fam0["SEX"] = [1]*half_pop_size+[2]*half_pop_size
    fam0["PHEN"] = np.hstack((male_phenotypes[0], female_phenotypes[0]))
    iids = list(range(base_iid, base_iid+2*half_pop_size))
    base_iid += 2*half_pop_size
    fam1["IID"] = iids
    fam1["PID"] = np.hstack((male_father_ranks[1]+1, female_father_ranks[1]+1))
    fam1["MID"] = half_pop_size+np.hstack((male_mother_ranks[1]+1, female_mother_ranks[1]+1))
    fam1["SEX"] = [1]*half_pop_size+[2]*half_pop_size
    fam1["PHEN"] = np.hstack((male_phenotypes[1], female_phenotypes[1]))
    fam1["FID"] = fam1["PID"]
    mothers_in_order = np.argsort(fam1["MID"].unique())
    fam0["FID"] = np.hstack((fam0["IID"][:half_pop_size], fam1["PID"][mothers_in_order]))

    output_address0 = output_address.replace("&", str(0))
    output_address1 = output_address.replace("&", str(1))
    output_bim.to_csv(output_address0+".bim", header=False, sep="\t", index=False)
    output_bim.to_csv(output_address1+".bim", header=False, sep="\t", index=False)
    fam0.to_csv(output_address0+".fam", header=False, sep=" ", index=False)
    fam1.to_csv(output_address1+".fam", header=False, sep=" ", index=False)

    hf.close()

def write_data(data, chr, start, end, gen):
    address = f"/disk/genetics4/ukb/alextisyoung/haplotypes/sim_genotypes/QCed/cache/loaded_data_from{chr}_to{chr+1}_start{start}_end{end}_gen{gen}.pickle"
    logging.info("++++++++++++++++++++++++++++")
    logging.info(address)
    with open(address, "wb") as f:
        pickle.dump(data, f)


def prepare_data(chr_range, bgen_address = "/disk/genetics4/ukb/alextisyoung/haplotypes/sim_genotypes/QCed/chr_~", start=0, end=None, gen = 0):
    from_chr = min(chr_range)
    to_chr = max(chr_range)
    #TODO fix address
    # if gen!=0:
    #     loaded_address = f"/disk/genetics4/ukb/alextisyoung/haplotypes/sim_genotypes/QCed/cache/loaded_data_from{from_chr}_to{to_chr+1}_start{start}_end{end}_gen{gen}.pickle"
    #     if os.path.isfile(loaded_address):
    #         logging.info("cache exists")
    #         with h5py.File(address, 'r') as hf:
    #             phased_gts = np.array(hf["phased_gts"])
    #             map_list = np.array(hf["map_list"]).tolist()
    #             chrom_lengths = list(hf["chrom_lengths"])
    #             for i in range(len(map_list)):
    #                 map_list[i] = map_list[i][:chrom_lengths[i]]
    #         return phased_gts, map_list, chrom_lengths
    #     else:
    #         logging.info("================================================")
    #         logging.info(loaded_address)
    #         raise Exception("no data for that gen")
    name_prefix_len = len(bgen_address.split("/")[-1])
    loaded_address = bgen_address[:-name_prefix_len]+f"cache/loaded_data_from{from_chr}_to{to_chr}_start{start}_end{end}.hdf5"
    logging.info("checking for loaded in "+ loaded_address)
    if os.path.isfile(loaded_address):
        logging.info("cache exists")
        with h5py.File(loaded_address, 'r') as hf:
            phased_gts = np.array(hf["phased_gts"])
            map_list = np.array(hf["map_list"]).tolist()
            chrom_lengths = list(hf["chrom_lengths"])
            # bim = pd.DataFrame(hf["bim"])
            bim_values = np.array(hf["bim_values"])
            bim_columns = np.array(hf["bim_columns"])
            bim = pd.DataFrame(bim_values, columns=bim_columns)
            for i in range(len(map_list)):
                map_list[i] = map_list[i][:chrom_lengths[i]]
        logging.info("loaded")
        return phased_gts, map_list, chrom_lengths, bim
    chr_starts = np.array(chr_range)
    chrom_lengths =[]
    whole_genotype = []
    poss = []
    bims = []
    for chrom in chr_range:
        logging.info("reading chromosome " + str(chrom))
        bgen = read_bgen(bgen_address.replace("~", str(chrom))+".bgen", verbose=False)
        bgen_bim = bgen["variants"].compute().reset_index()
        bgen_bim = bgen_bim.rename(columns={"chrom":"Chr", "pos":"coordinate", "index":"base_pair"})
        variants_number = bgen_bim.shape[0]
        this_end = end
        if this_end is None:
            this_end = variants_number
        bims.append(bgen_bim[start:this_end])
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
            genomes = np.array([(-1, -1)]*pop_size, dtype=np.int8)
            genomes[probs[:,0] > 0.99, 0] = 1
            genomes[probs[:,1] > 0.99, 0] = 0
            genomes[probs[:,2] > 0.99, 1] = 1
            genomes[probs[:,3] > 0.99, 1] = 0
            chromosome_genotype.append(genomes)
        whole_genotype = whole_genotype + chromosome_genotype
    complete_bim = pd.concat(bims)
    bim_values = complete_bim.values.astype("S")
    bim_columns = complete_bim.columns.values.astype("S")
    bim = pd.DataFrame(bim_values, columns = bim_columns)
    phased_gts = np.array(whole_genotype, dtype=np.int8)#.transpose(1,0,2)
    logging.info("phased_gts.shape" + str(phased_gts.shape))
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
    with h5py.File(loaded_address, 'w') as hf:
        logging.info("wrote to" + loaded_address)
        hf.create_dataset("phased_gts", phased_gts.shape, chunks = True, compression = "lzf", data = phased_gts)    
        matrix_map_list = np.zeros((len(map_list), max(chrom_lengths)))
        for i in range(len(map_list)):
            matrix_map_list[i, :chrom_lengths[i]] = map_list[i]
        hf["map_list"] = matrix_map_list
        hf["chrom_lengths"] = chrom_lengths
        hf["bim_values"] = bim_values
        hf["bim_columns"] = bim_columns
    return phased_gts, map_list, chrom_lengths, bim
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

def recombination(population, map_list, chrom_lengths, chunks = 100, processes = 10):
    with Pool(processes) as p:
        pop_size = population.shape[1]
        bin_size = (pop_size-1)//chunks+1 
        pop_chunks = [population[:, i*bin_size: min((i+1)*bin_size, pop_size), :] for i in range(chunks)]
        args = [(pop_chunks[i], map_list, chrom_lengths) for i in range(chunks)]
        results = p.starmap(recombination_chunk, args)
    assembled_pop = np.hstack(results)
    return assembled_pop

def recombination_chunk(population, map_list, chrom_lengths):
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
            causal_mask,
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
    males_sum = [None]*length
    females_sum = [None]*length
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
    males[0] = phased_gts[:, [i%2==0 for i in range(pop_size)], :].astype(np.int8)
    females[0] = phased_gts[:, [i%2==1 for i in range(pop_size)], :].astype(np.int8)
    males_sum[0] = np.sum(males[0], 2).astype(float)
    females_sum[0] = np.sum(females[0], 2).astype(float)
    male_direct[0, :] = a@males_sum[0]
    female_direct[0, :] = a@females_sum[0]
    male_phenotypes[0, :] = male_direct[0, :] + np.random. normal(0, sqrt(ve), (1, pop_size//2))
    female_phenotypes[0, :] = female_direct[0, :] + np.random.normal(0, sqrt(ve), (1, pop_size//2))
    vy[0] = np.var(np.hstack((male_phenotypes[0, :], female_phenotypes[0, :])))
    v_direct[0] = np.var(np.hstack((male_direct[0, :], female_direct[0, :])))
    heritablity[0] = v_direct[0]/vy[0]
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
    orriginal_p=p
    for i in tqdm(range(1, length)):
        if i>1:
            p = orriginal_p
        else:
            p = 0.000001
        if i >= latest_mem:
            males[i-latest_mem] = None
            females[i-latest_mem] = None
            males_sum[i-latest_mem] = None
            females_sum[i-latest_mem] = None
        logging.info(f"generation {i} start mating")
        delta = 0.9
        dif = 1
        while dif>=0:
            delta += 0.1
            father_phenotype = male_phenotypes[i-1, :]
            mother_phenotype = female_phenotypes[i-1, :]
            father = males[i-1]
            mother = females[i-1]
            logging.info(f"adding noise ...")
            noisy_father_phenotype = father_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=father_phenotype.shape)
            noisy_mother_phenotype = mother_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=mother_phenotype.shape)
            logging.info(f"sorting ...")
            sorted_to_initial_ranks_male = np.argsort(noisy_father_phenotype)
            sorted_to_initial_ranks_female = np.argsort(noisy_mother_phenotype)
            logging.info(f"selecting ...")
            father = father[:, sorted_to_initial_ranks_male,:]
            mother = mother[:, sorted_to_initial_ranks_female,:]
            father_sum = males_sum[i-1][:, sorted_to_initial_ranks_male]
            mother_sum = females_sum[i-1][:, sorted_to_initial_ranks_female]
            father_phenotype_sorted = father_phenotype[sorted_to_initial_ranks_male]
            mother_phenotype_sorted = mother_phenotype[sorted_to_initial_ranks_female]
            logging.info(f"computing corr ...")
            temp_cor = np.corrcoef(father_phenotype_sorted, mother_phenotype_sorted)
            dif = temp_cor[0,1]-p
            logging.info(f"computing corr done")
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
            logging.info(f"sorting ...")
            noisy_father_phenotype = father_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=father_phenotype.shape)
            noisy_mother_phenotype = mother_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=mother_phenotype.shape)
            logging.info(f"sorting ...")
            sorted_to_initial_ranks_male = np.argsort(noisy_father_phenotype)
            sorted_to_initial_ranks_female = np.argsort(noisy_mother_phenotype)
            logging.info(f"selecting ...")
            father = father[:, sorted_to_initial_ranks_male, :]
            mother = mother[:, sorted_to_initial_ranks_female, :]
            father_sum = males_sum[i-1][:, sorted_to_initial_ranks_male]
            mother_sum = females_sum[i-1][:, sorted_to_initial_ranks_female]
            father_phenotype_sorted = father_phenotype[sorted_to_initial_ranks_male]
            mother_phenotype_sorted = mother_phenotype[sorted_to_initial_ranks_female]
            logging.info(f"computing corr ...")
            temp_cor = np.corrcoef(father_phenotype_sorted, mother_phenotype_sorted)
            dif = temp_cor[0,1]-p
            logging.info(f"computing corr done")
        logging.info(f"generation {i} sorted")                
        deltas[i] = delta
        corrs[i-1] = temp_cor[0,1]
        # corrs_inbreeding[i-1] = np.corrcoef(np.sum(father[causal_mask,:,:], axis=2).reshape((1, -1)), np.sum(mother[causal_mask,:,:], axis=2).reshape((1, -1)))[0,1]
        # corrs_causal[i-1] = np.corrcoef(np.sum(father[causal_mask,:,:], axis=2).reshape((1, -1)), np.sum(mother[causal_mask,:,:], axis=2).reshape((1, -1)))[0,1]
        logging.info(f"figuring parents")
        orders = np.random.permutation(range(father.shape[1]))
        son_fams = orders[[i%2==0 for i in range(len(orders))]]
        daughter_fams = orders[[i%2!=0 for i in range(len(orders))]]
        son_fathers = father[:, son_fams, :]
        son_fathers_sum = father_sum[:, son_fams]
        son_mothers = mother[:, son_fams, :]
        son_mothers_sum = mother_sum[:, son_fams]
        daughter_fathers = father[:, daughter_fams, :]
        daughter_fathers_sum = father_sum[:, daughter_fams]
        daughter_mothers = mother[:, daughter_fams, :]
        daughter_mothers_sum = mother_sum[:, daughter_fams,]
        logging.info(f"mate ...")
        son1 = mate(son_fathers, son_mothers, map_list, chrom_lengths)
        son2 = mate(son_fathers, son_mothers, map_list, chrom_lengths)
        daughter1 = mate(daughter_fathers, daughter_mothers, map_list, chrom_lengths)
        daughter2 = mate(daughter_fathers, daughter_mothers, map_list, chrom_lengths)
        logging.info(f"mating done")
        males[i] = np.hstack((son1, son2))
        females[i] = np.hstack((daughter1, daughter2))
        males_sum[i] = np.sum(males[i], 2)
        females_sum[i] = np.sum(females[i], 2)
        male_parent_sorted_ranks = np.hstack((son_fams, son_fams))
        male_father_ranks[i] = np.array([sorted_to_initial_ranks_male[male_parent_sorted_ranks[j]] for j in range(pop_size//2)])
        male_mother_ranks[i] = np.array([sorted_to_initial_ranks_female[male_parent_sorted_ranks[j]] for j in range(pop_size//2)])

        female_parent_sorted_ranks = np.hstack((daughter_fams, daughter_fams))
        female_father_ranks[i] = np.array([sorted_to_initial_ranks_male[female_parent_sorted_ranks[j]] for j in range(pop_size//2)])
        female_mother_ranks[i] = np.array([sorted_to_initial_ranks_female[female_parent_sorted_ranks[j]] for j in range(pop_size//2)])
        logging.info(f"ranks computed")
        if statistics_function is not None:
            statistics_function(i, data, statistics)
        logging.info(f"male direct ...")
        male_direct[i, :] = a@males_sum[i]
        logging.info(f"male direct done")
        logging.info(f"male indirect ...")
        male_indirect[i, :] = b@(np.hstack((son_fathers_sum, son_fathers_sum)) + np.hstack((son_mothers_sum , son_mothers_sum)))
        logging.info(f"male indirect ...")
        
        logging.info(f"female direct ...")
        female_direct[i, :] = a@females_sum[i]
        logging.info(f"female direct done")
        logging.info(f"female indirect ...")
        female_indirect[i, :] = b@(np.hstack((daughter_fathers_sum, daughter_fathers_sum)) + np.hstack((daughter_mothers_sum,  daughter_mothers_sum)))
        logging.info(f"female indirect done")
        logging.info(f"assembling phenotype")
        male_phenotypes[i, :] = male_direct[i, :] + male_indirect[i, :] + np.random. normal(0, sqrt(ve), (1, pop_size//2))
        female_phenotypes[i, :] = female_direct[i, :] + female_indirect[i, :] + np.random.normal(0, sqrt(ve), (1, pop_size//2))
        logging.info(f"assembling phenotype done")

        logging.info(f"computing vy ...")        
        vy[i] = np.var(np.hstack((male_phenotypes[i, :], female_phenotypes[i, :])))
        logging.info(f"computing vy done")
        logging.info(f"computing direct indirect ...")
        direct_indirect_cov_mat = np.cov(np.hstack((male_direct[i, :], female_direct[i, :])).flatten(), np.hstack((male_indirect[i, :], female_indirect[i, :])).flatten())
        logging.info(f"computing direct indirect done")
        v_direct[i] = direct_indirect_cov_mat[0,0]
        v_indirect[i] = direct_indirect_cov_mat[1,1]
        cov_direct_indirect[i] = direct_indirect_cov_mat[0, 1]
        heritablity[i] = v_direct[i]/vy[i]
        father_direct = np.hstack((male_direct[i-1, female_father_ranks[i]], male_direct[i-1, male_father_ranks[i]]))
        mother_direct = np.hstack((female_direct[i-1, female_mother_ranks[i]], female_direct[i-1, male_mother_ranks[i]]))
        father_indirect = np.hstack((male_indirect[i-1, female_father_ranks[i]], male_indirect[i-1, male_father_ranks[i]]))
        mother_indirect = np.hstack((female_indirect[i-1, female_mother_ranks[i]], female_indirect[i-1, male_mother_ranks[i]]))
        logging.info(f"computing parental direct indirect ...")
        cov_parental_direct[i] = np.cov(father_direct, mother_direct)[0,1]
        cov_parental_indirect[i] = np.cov(father_indirect, mother_direct)[0,1]
        cov_parental_direct_indirect[i] = np.cov(np.hstack((father_direct, mother_direct)), np.hstack((mother_indirect, father_indirect)))[0,1]
        logging.info(f"computing parental direct indirect done")
        logging.info(f"direct indirect statistics done")

    
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


# def simulate_batch(
#             length,
#             causal_genes,
#             chr_range,
#             pop_size, number_of_genes,
#             a,
#             b,
#             p,
#             ve,
#             latest_mem,            
#             statistics_function = None,
#             statistics = {},
#             force_p=True,            
#             ):

#     if latest_mem is None:
#         latest_mem = length    
    
#     female_father_ranks = np.zeros((length, pop_size//2)).astype(int)
#     female_mother_ranks = np.zeros((length, pop_size//2)).astype(int)
#     male_father_ranks = np.zeros((length, pop_size//2)).astype(int)
#     male_mother_ranks = np.zeros((length, pop_size//2)).astype(int)
#     male_phenotypes = np.array([[None]*(pop_size//2)]*length, dtype=float)
#     female_phenotypes = np.array([[None]*(pop_size//2)]*length, dtype=float)
#     male_direct = np.array([[0]*(pop_size//2)]*length, dtype=float)
#     male_indirect = np.array([[0]*(pop_size//2)]*length, dtype=float)
#     female_direct = np.array([[0]*(pop_size//2)]*length, dtype=float)
#     female_indirect = np.array([[0]*(pop_size//2)]*length, dtype=float)
#     male_indirect_buffer = np.array([[0]*(pop_size//2)]*length, dtype=float)
#     female_indirect_buffer = np.array([[0]*(pop_size//2)]*length, dtype=float)
#     cov_parental_direct = np.array([None]*length, dtype=float)
#     cov_parental_indirect = np.array([None]*length, dtype=float)
#     cov_parental_direct_indirect = np.array([None]*length, dtype=float)
#     vy = np.array([None]*length, dtype=float)
#     v_direct = np.array([None]*length, dtype=float)
#     v_indirect = np.array([None]*length, dtype=float)
#     cov_direct_indirect = np.array([None]*length, dtype=float)
#     heritablity = np.array([None]*length, dtype=float)
#     corrs = np.array([None]*length, dtype=float)
#     corrs_original = np.array([None]*length, dtype=float)
#     corrs_inbreeding = np.array([None]*length, dtype=float)
#     corrs_causal = np.array([None]*length, dtype=float)
#     deltas = np.array([None]*length, dtype=float)
#     #TODO figure this out
#     size = 2
#     for chr in chr_range:
#         phased_gts, map_list, chrom_lengths = prepare_data([chr], start=0, end=50)
#         males = phased_gts[:, :pop_size//2, :].astype(float)
#         females = phased_gts[:, pop_size//2:, :].astype(float)
#         male_direct[0, :] += (a[:, (chr-1)*size: chr*size]@np.sum(males,2)[:size, :])[0]
#         female_direct[0, :] += (a[:, (chr-1)*size: chr*size]@np.sum(females,2)[:size, :])[0]
#         # write indirect of everyone on the next gen
#         male_indirect_buffer[0, :] += (b[:, (chr-1)*size: chr*size]@np.sum(males,2)[:size, :])[0]
#         female_indirect_buffer[0, :] += (b[:, (chr-1)*size: chr*size]@np.sum(females,2)[:size, :])[0]
    

        
#     male_phenotypes[0, :] = male_direct[0, :] + np.random. normal(0, sqrt(ve), (1, pop_size//2))
#     female_phenotypes[0, :] = female_direct[0, :] + np.random.normal(0, sqrt(ve), (1, pop_size//2))
#     vy[0] = np.var(np.hstack((male_phenotypes[0, :], female_phenotypes[0, :])))
#     v_direct[0] = np.var(np.hstack((male_direct[0, :], female_direct[0, :])))
#     heritablity[0] = v_direct[0]/vy[0]

#     for i in tqdm(range(1, length)):
#         delta = 0.9
#         dif = 1
#         while dif>=0:
#             delta += 0.1
#             father_phenotype = male_phenotypes[i-1, :]
#             mother_phenotype = female_phenotypes[i-1, :]
#             noisy_father_phenotype = father_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=father_phenotype.shape)
#             noisy_mother_phenotype = mother_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=mother_phenotype.shape)
#             noisy_father_phenotype_ranks = np.argsort(noisy_father_phenotype)
#             noisy_mother_phenotype_ranks = np.argsort(noisy_mother_phenotype)
#             father_phenotype_sorted = father_phenotype[noisy_father_phenotype_ranks]
#             mother_phenotype_sorted = mother_phenotype[noisy_mother_phenotype_ranks]
#             temp_cor = np.corrcoef(father_phenotype_sorted, mother_phenotype_sorted)
#             dif = temp_cor[0,1]-p
#             if delta == 1:
#                 corrs_original[i] = temp_cor[0,1]
#             if not force_p:
#                 break
#         while dif<0 and force_p:
#             delta -= 0.01
#             father_phenotype = male_phenotypes[i-1, :]
#             mother_phenotype = female_phenotypes[i-1, :]
#             noisy_father_phenotype = father_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=father_phenotype.shape)
#             noisy_mother_phenotype = mother_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]*delta), size=mother_phenotype.shape)
#             noisy_father_phenotype_ranks = np.argsort(noisy_father_phenotype)
#             noisy_mother_phenotype_ranks = np.argsort(noisy_mother_phenotype)
#             father_phenotype_sorted = father_phenotype[noisy_father_phenotype_ranks]
#             mother_phenotype_sorted = mother_phenotype[noisy_mother_phenotype_ranks]
#             temp_cor = np.corrcoef(father_phenotype_sorted, mother_phenotype_sorted)
#             dif = temp_cor[0,1]-p



#         sorted_to_initial_ranks_male = np.argsort(noisy_father_phenotype)
#         sorted_to_initial_ranks_female = np.argsort(noisy_mother_phenotype)
#         deltas[i] = delta
#         corrs[i-1] = temp_cor[0,1]
#         # corrs_inbreeding[i-1] = np.corrcoef(np.sum(father[causal_genes:,:,:], axis=2).reshape((1, -1)), np.sum(mother[causal_genes:,:,:], axis=2).reshape((1, -1)))[0,1]
#         # corrs_causal[i-1] = np.corrcoef(np.sum(father[:causal_genes,:,:], axis=2).reshape((1, -1)), np.sum(mother[:causal_genes,:,:], axis=2).reshape((1, -1)))[0,1]
    
#         orders = np.random.permutation(range(pop_size//2))
#         son_fams = orders[[i%2==0 for i in range(len(orders))]]
#         daughter_fams = orders[[i%2!=0 for i in range(len(orders))]]
#         male_parent_sorted_ranks = np.hstack((son_fams, son_fams))
#         male_father_ranks = np.array([sorted_to_initial_ranks_male[male_parent_sorted_ranks[j]] for j in range(pop_size//2)])
#         male_mother_ranks = np.array([sorted_to_initial_ranks_female[male_parent_sorted_ranks[j]] for j in range(pop_size//2)])
#         female_parent_sorted_ranks = np.hstack((daughter_fams, daughter_fams))
#         female_father_ranks = np.array([sorted_to_initial_ranks_male[female_parent_sorted_ranks[j]] for j in range(pop_size//2)])
#         female_mother_ranks = np.array([sorted_to_initial_ranks_female[female_parent_sorted_ranks[j]] for j in range(pop_size//2)])

#         for chr in chr_range:
#             start = 0
#             end = 2
#             size = end-start
#             phased_gts, map_list, chrom_lengths = prepare_data([chr], start=start, end=end, gen=i-1)
#             males = phased_gts[:, :pop_size, :].astype(float)
#             females = phased_gts[:, :pop_size, :].astype(float)
#             fathers = males[:, noisy_father_phenotype_ranks]
#             mothers = females[:, noisy_mother_phenotype_ranks]
#             son_fathers = fathers[:, son_fams, :]
#             son_mothers = mothers[:, son_fams, :]
#             daughter_fathers = fathers[:, daughter_fams, :]
#             daughter_mothers = mothers[:, daughter_fams, :]
#             son1 = mate(son_fathers, son_mothers, map_list, chrom_lengths)
#             son2 = mate(son_fathers, son_mothers, map_list, chrom_lengths)
#             daughter1 = mate(daughter_fathers, daughter_mothers, map_list, chrom_lengths)
#             daughter2 = mate(daughter_fathers, daughter_mothers, map_list, chrom_lengths)
#             sons = np.hstack((son1, son2))
#             daughters = np.hstack((daughter1, daughter2))
#             next_gen = np.hstack((sons, daughters))
#             write_data((next_gen, map_list, chrom_lengths), chr, start, end, i)
        
#             male_direct[i, :] += (a[:, (chr-1)*size: chr*size]@np.sum(sons,2)[:size, :])[0]
#             female_direct[i, :] += (a[:, (chr-1)*size: chr*size]@np.sum(daughters,2)[:size, :])[0]
#             female_indirect[i, :] = male_indirect_buffer[i-1, female_father_ranks]+female_indirect_buffer[i-1, female_mother_ranks]
#             male_indirect[i, :] = male_indirect_buffer[i-1, male_father_ranks]+female_indirect_buffer[i-1, male_mother_ranks]
           
#             male_indirect_buffer[1, :] += (b[:, (chr-1)*size: chr*size]@np.sum(sons,2)[:size, :])[0]
#             male_indirect_buffer[1, :] += (b[:, (chr-1)*size: chr*size]@np.sum(daughters,2)[:size, :])[0]

#         male_phenotypes[i, :] = male_direct[i, :] + male_indirect[i, :] + np.random. normal(0, sqrt(ve), (1, pop_size//2))
#         female_phenotypes[i, :] = female_direct[i, :] + female_indirect[i, :] + np.random.normal(0, sqrt(ve), (1, pop_size//2))
#         vy[i] = np.var(np.hstack((male_phenotypes[i, :], female_phenotypes[i, :])))
#         direct_indirect_cov_mat = np.cov(np.hstack((male_direct[i, :], female_direct[i, :])).flatten(), np.hstack((male_indirect[i, :], female_indirect[i, :])).flatten())
#         v_direct[i] = direct_indirect_cov_mat[0,0]
#         v_indirect[i] = direct_indirect_cov_mat[1,1]
#         cov_direct_indirect[i] = direct_indirect_cov_mat[0, 1]
#         heritablity[i] = v_direct[i]/vy[i]
#         father_direct = np.hstack((male_direct[i-1, female_father_ranks[i]], male_direct[i-1, male_father_ranks[i]]))
#         mother_direct = np.hstack((female_direct[i-1, female_mother_ranks[i]], female_direct[i-1, male_mother_ranks[i]]))
#         father_indirect = np.hstack((male_indirect[i-1, female_father_ranks[i]], male_indirect[i-1, male_father_ranks[i]]))
#         mother_indirect = np.hstack((female_indirect[i-1, female_mother_ranks[i]], female_indirect[i-1, male_mother_ranks[i]]))
#         cov_parental_direct[i] = np.cov(father_direct, mother_direct)[0,1]
#         cov_parental_indirect[i] = np.cov(father_indirect, mother_direct)[0,1]
#         cov_parental_direct_indirect[i] = np.cov(np.hstack((father_direct, mother_direct)), np.hstack((mother_indirect, father_indirect)))[0,1]

    
#     return {"vy":vy, 
#             # "corrs":corrs,
#             # "corrs_causal":corrs_causal,
#             # "corrs_inbreeding":corrs_inbreeding,
#             "male_phenotypes":male_phenotypes,
#             "female_phenotypes":female_phenotypes,
#             "male_direct":male_direct,
#             "male_indirect":male_indirect,
#             "female_direct":female_direct,
#             "female_indirect":female_indirect,
#             "deltas":deltas,
#             # "males":np.array([gen.tolist() for gen in males if gen is not None]),
#             # "females":np.array([gen.tolist() for gen in females if gen is not None]),
#             "female_father_ranks": female_father_ranks,
#             "female_mother_ranks": female_mother_ranks,
#             "male_father_ranks": male_father_ranks,
#             "male_mother_ranks": male_mother_ranks,
#             "v_direct": v_direct,
#             "v_indirect": v_indirect,
#             "cov_direct_indirect": cov_direct_indirect,
#             "heritablity": heritablity,
#             "cov_parental_direct": cov_parental_direct,
#             "cov_parental_indirect": cov_parental_indirect,
#             "cov_parental_direct_indirect": cov_parental_direct_indirect,
#         }