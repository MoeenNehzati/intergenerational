from pop_sim_recom import simulate, prepare_data
import numpy as np
import h5py
import itertools
from multiprocessing import Pool
import argparse
import os
from utils import simulate_generations

def use_derivations(phased_gts, map_list, chrom_lengths, length, a, b, p, ve):
    vg0 = np.cov(np.sum(phased_gts, 2))
    # plt.imshow(vg0, cmap='hot', interpolation='nearest')
    cm_to_recom = lambda d:(1-np.exp(-2*d/100))/2
    c = np.zeros(vg0.shape)
    c[:,:] = 0.5
    start = 0
    for index, l in enumerate(chrom_lengths):
        for i in range(l):
            for j in range(l):
                cm = np.abs(map_list[index][i] - map_list[index][j])
                recom = cm_to_recom(cm)
                c[start+i, start+j] = recom
        start += l    
    expected_vg, expected_vy, expected_m, expected_h2, expected_sib_cov = simulate_generations(length, vg0, c, a, b, p, ve)
    return expected_vg, expected_vy, expected_m, expected_h2, expected_sib_cov

def f(cov, length, causal_genes, phased_gts, map_list, chrom_lengths, p, ve, mem, run, force_p=True):
    np.random.seed()
    name = f"from_chr{from_chr}_to_chr{to_chr}_start{start}_end{end}_run{run}_p{round(p, 2)}_ab_corr{round(cov[0][1]/np.sqrt(cov[1][1]+0.000001), 2)}_vb{round(cov[1][1], 2)}".replace(".","-")
    if os.path.isfile(f"outputs/{name}.hdf5"):
        print(f"outputs/{name}.hdf5 exists")
        return None, None
    else:
        print(f"outputs/{name}.hdf5 simulating...")
    ab = np.random.multivariate_normal([0,0], cov, number_of_the_genes)
    a = ab[:,[0]].T
    b = ab[:,[1]].T
    temp_var = np.std(a@np.sum(phased_gts, axis=2))
    a = a/temp_var
    b = b/temp_var
    result = simulate(length, causal_genes, phased_gts, map_list, chrom_lengths, a, b, p, ve, mem, force_p=True)
    expected_vg, expected_vy, expected_m, expected_h2, expected_sib_cov = use_derivations(phased_gts, map_list, chrom_lengths, length, a, b, p, ve)
    result["ab_cov"] = np.array(cov)
    result["p"] = p
    result["ve"] = ve
    result["start"] = start
    result["end"] = end
    result["from_chr"] = from_chr
    result["to_chr"] = to_chr
    result["runs"] = runs
    result["causal_genes"] = causal_genes    
    result["a"] = a
    result["b"] = b
    result["name"] = bytes(name, "ascii")
    result["expected_vg"] = expected_vg
    result["expected_vy"] = expected_vy
    result["expected_m"] = expected_m
    result["expected_h2"] = expected_h2
    result["expected_sib_cov"] = expected_sib_cov
    with h5py.File(f"outputs/{name}.hdf5",'w') as f:
        for key, val in result.items():
            f[key] = np.array(val)
    return name, result

parser = argparse.ArgumentParser()
parser.add_argument('-start', type=int, help='start for each chromosome', default = 0)
parser.add_argument('-end', type=int, help='end for each chromosome', default = None)
parser.add_argument('-fromchr', type=int, help='starting chromosome')
parser.add_argument('-tochr', type=int, help='until chromosome')
parser.add_argument('-processes', type=int, help='number of processes', default = 1)
parser.add_argument('-runs', type=int, help='number of simulations per case', default = 1)
args=parser.parse_args()
start = args.start
end = args.end
from_chr = args.fromchr
to_chr = args.tochr
processes = args.processes
runs = args.runs

np.random.seed(11523)
phased_gts, map_list, chrom_lengths = prepare_data(range(from_chr, to_chr), start = start, end = end)
number_of_the_genes = phased_gts.shape[0]
pop_size = phased_gts.shape[1]
fs = np.mean(np.sum(phased_gts, axis=2)/2, axis=1)
causal_genes = number_of_the_genes
length = 30
ve = 1.
mem = 2

only_direct_covs = [[1, 0], [0, 0]] #only direct
prefect_cor_covs = [[1, 0.5], [0.5, 0.25]] #perfectly correlated parental and direct
imprefect_cor_covs = [[1, 0.25], [0.25, 0.25]] # somewhat correlated
no_cor_covs = [[1, 0], [0, 0.25]] # somewhat correlated
covs = [only_direct_covs, prefect_cor_covs, imprefect_cor_covs, no_cor_covs]
ps = [0.000001, 0.5]
all_results = {}

results = []
with Pool(processes) as pool:
    args = [(cov, length, causal_genes, phased_gts, map_list, chrom_lengths, p, ve, mem, run) for cov, p, run in itertools.product(covs, ps, range(runs))]
    results = pool.starmap(f, args)





# import h5py
# from matplotlib import pyplot as plt
# import numpy as np
# from tqdm import tqdm
# from utils import simulate_generations
# import os
# addresses = [
#     "outputs/from_chr1_to_chr23_start0_end50_run&_p0-0_ab_corr0-0_vb0-25.hdf5",
#     "outputs/from_chr1_to_chr23_start0_end50_run&_p0-0_ab_corr0-0_vb0.hdf5",
#     "outputs/from_chr1_to_chr23_start0_end50_run&_p0-0_ab_corr0-5_vb0-25.hdf5",
#     "outputs/from_chr1_to_chr23_start0_end50_run&_p0-0_ab_corr1-0_vb0-25.hdf5",
#     "outputs/from_chr1_to_chr23_start0_end50_run&_p0-5_ab_corr0-0_vb0-25.hdf5",
#     "outputs/from_chr1_to_chr23_start0_end50_run&_p0-5_ab_corr0-0_vb0.hdf5",
#     "outputs/from_chr1_to_chr23_start0_end50_run&_p0-5_ab_corr0-5_vb0-25.hdf5",
#     "outputs/from_chr1_to_chr23_start0_end50_run&_p0-5_ab_corr1-0_vb0-25.hdf5",
# ]
# length = 30
# runs=10

# plt.clf()
# for address_widlcard in tqdm(addresses):
#     ys = np.zeros((runs, length-2))
#     for i in range(runs):
#         address = address_widlcard.replace("&", str(i))
#         print(address)
#         if not os.path.isfile(address):
#             print(address, "pass")
#             ys[i, :] = np.nan
#             continue
#         with h5py.File(address, 'r') as hf:
#             # name = address.split("/")[1].split(".")[0]
#             vy = np.array(hf["vy"])
#             p = np.array(hf["p"])
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             y = vy[2:]/vy[2]
#             ys[i, :] = y
#     y = np.nanmean(ys, 0)
#     plt.plot(x,y, label=f"vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, p={p}")


# plt.title(f"Vy (normalized on the first gen) for {runs} runs")
# plt.legend()
# plt.xlabel("gen")
# plt.ylabel("normalized vy")
# plt.savefig(f"outputs/normalized_vy_from_chr1_to_chr23_start0_end50_runs{runs}")



# plt.clf()
# for address_widlcard in tqdm(addresses):
#     ys = np.zeros((runs, length-2))
#     for i in range(runs):
#         address = address_widlcard.replace("&", str(i))
#         print(address)
#         if not os.path.isfile(address):
#             print(address, "pass")
#             ys[i, :] = np.nan
#             continue
#         with h5py.File(address, 'r') as hf:
#             # name = address.split("/")[1].split(".")[0]
#             h2 = np.array(hf["heritablity"])
#             p = np.array(hf["p"])
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             y = h2[2:]/h2[2]
#             ys[i, :] = y
#     y = np.nanmean(ys, 0)
#     plt.plot(x,y, label=f"vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, p={p}")


# plt.title(f"h2(normalized on the first gen) for {runs} runs")
# plt.legend()
# plt.xlabel("gen")
# plt.ylabel("normalized h2")
# plt.savefig(f"outputs/normalized_h2_from_chr1_to_chr23_start0_end50_runs{runs}")


# plt.clf()
# for address_widlcard in tqdm(addresses):
#     ys = np.zeros((runs, length-2))
#     for i in range(runs):
#         address = address_widlcard.replace("&", str(i))
#         print(address)
#         if not os.path.isfile(address):
#             print(address, "pass")
#             ys[i, :] = np.nan
#             continue
#         with h5py.File(address, 'r') as hf:
#             # name = address.split("/")[1].split(".")[0]
#             h2 = np.array(hf["heritablity"])
#             p = np.array(hf["p"])
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             y = h2[2:]
#             ys[i, :] = y
#     y = np.nanmean(ys, 0)
#     plt.plot(x,y, label=f"vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, p={p}")


# plt.title(f"h2 for {runs} runs")
# plt.legend()
# plt.xlabel("gen")
# plt.ylabel("h2")
# plt.savefig(f"outputs/h2_from_chr1_to_chr23_start0_end50_runs{runs}")


# plt.clf()
# for address_widlcard in tqdm(addresses):
#     ys = np.zeros((runs, length-2))
#     for i in range(runs):
#         address = address_widlcard.replace("&", str(i))
#         print(address)
#         if not os.path.isfile(address):
#             print(address, "pass")
#             ys[i, :] = np.nan
#             continue
#         with h5py.File(address, 'r') as hf:
#             # name = address.split("/")[1].split(".")[0]
#             vy = np.array(hf["vy"])
#             p = np.array(hf["p"])
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             y = vy[2:]
#             ys[i, :] = y
#     y = np.nanmean(ys, 0)
#     plt.plot(x,y, label=f"vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, p={p}")


# plt.title(f"Vy for {runs} runs")
# plt.legend()
# plt.xlabel("gen")
# plt.ylabel("Vy")
# plt.savefig(f"outputs/vy_from_chr1_to_chr23_start0_end50_runs{runs}")


# plt.clf()
# for address_widlcard in tqdm(addresses[4:]):
#     sim = np.zeros((runs, length-2))
#     expected = np.zeros((runs, length-2))
#     for i in range(runs):
#         address = address_widlcard.replace("&", str(i))
#         if not os.path.isfile(address):
#             print(address, "pass")
#             sim[i, :] = np.nan
#             expected[i, :] = np.nan
#             continue
#         with h5py.File(address, 'r') as hf:
#             vy = np.array(hf["vy"])
#             vy_expected = np.array(hf["expected_vy"])            
#             ve = np.array(hf["ve"])
#             p = np.array(hf["p"])
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             sim[i,:] = vy[2:]
#             expected[i,:] = vy_expected[2:]
#     sim_avg = np.nanmean(sim, 0)
#     expected_avg = np.nanmean(expected, 0)
#     plt.plot(x, sim_avg, label=f"simulation vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, p={p}")
#     plt.plot(x, expected_avg, label=f" expected vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, p={p}")

# plt.title(f"Average Vy from simulation and derivations for {runs} runs")
# plt.legend()
# plt.xlabel("gen")
# plt.ylabel("Vy")
# plt.savefig(f"outputs/vy_expected_derivation_from_chr1_to_chr23_start0_end50_runs{runs}")


# plt.clf()
# for address_widlcard in tqdm(addresses[4:]):
#     sim = np.zeros((runs, length-2))
#     expected = np.zeros((runs, length-2))
#     for i in range(runs):
#         address = address_widlcard.replace("&", str(i))
#         if not os.path.isfile(address):
#             print(address, "pass")
#             sim[i, :] = np.nan
#             expected[i, :] = np.nan
#             continue
#         with h5py.File(address, 'r') as hf:
#             h = np.array(hf["heritablity"])
#             h_expected = np.array(hf["heritablity"])            
#             ve = np.array(hf["ve"])
#             p = np.array(hf["p"])
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             sim[i,:] = h[2:]
#             expected[i,:] = h_expected[2:]
#     sim_avg = np.nanmean(sim, 0)
#     expected_avg = np.nanmean(expected, 0)
#     plt.plot(x, sim_avg, label=f"simulation vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, p={p}")
#     plt.plot(x, expected_avg, label=f" expected vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, p={p}")

# plt.title(f"Average h2 from simulation and derivations for {runs} runs")
# plt.legend()
# plt.xlabel("gen")
# plt.ylabel("Vy")
# plt.savefig(f"outputs/h2_expected_derivation_from_chr1_to_chr23_start0_end50_runs{runs}")




# plt.clf()
# for address_widlcard in tqdm(addresses):
#     ys = np.zeros((runs, length-2))
#     for i in range(runs):
#         address = address_widlcard.replace("&", str(i))
#         print(address)
#         if not os.path.isfile(address):
#             print(address, "pass")
#             ys[i, :] = np.nan
#             continue
#         with h5py.File(address, 'r') as hf:
#             # name = address.split("/")[1].split(".")[0]
#             vy = np.array(hf["vy"])
#             p = np.array(hf["p"])
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-3)
#             ys[i, :] = vy[2:]#(vy[3:]-vy[2:-1])/vy[2:-1]
#     y = np.nanmean(ys, 0)
#     y = y[1:]-y[:-1]
#     plt.plot(x,y, label=f"vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, p={p}")


# plt.title(f"Convergence speed for {runs} runs")
# plt.legend()
# plt.xlabel("gen")
# plt.ylabel("proportional change in Vy")
# plt.savefig(f"outputs/convergence_speed_from_chr1_to_chr23_start0_end50_runs{runs}")




# ====================================================


# plt.clf()
# for address_widlcard in tqdm(addresses):
#     ys = np.zeros((runs, length-2))
#     for i in range(runs):
#         address = address_widlcard.replace("&", str(i))        
#         with h5py.File(address, 'r') as hf:
#             vy = np.array(hf["vy"])
#             vy_expected = np.array(hf["expected_vy"])            
#             ve = np.array(hf["ve"])
#             p = np.array(hf["p"])
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             y = (vy_expected[2:]-vy[2:])/vy_expected[2:]
#             ys[i, :] = y
#     y = np.mean(ys, 0)
#     plt.plot(x,y, label=f"vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, p={p}")

# plt.title(f"(expected_vy-vy)/expected_vy for {runs} runs")
# plt.legend()
# plt.xlabel("gen")
# plt.ylabel("proportional difference")
# plt.savefig(f"outputs/proportional_difference_vy_from_chr1_to_chr23_start0_end50_runs{runs}")


# plt.clf()
# for address_widlcard in tqdm(addresses):
#     ys = np.zeros((runs, length-2))
#     for i in range(runs):
#         address = address_widlcard.replace("&", str(i))        
#         with h5py.File(address, 'r') as hf:
#             vy = np.array(hf["vy"])
#             vy_expected = np.array(hf["expected_vy"])            
#             ve = np.array(hf["ve"])
#             p = np.array(hf["p"])
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             y = (vy_expected[2:]-vy[2:])
#             ys[i, :] = y
#     y = np.mean(ys, 0)
#     plt.plot(x,y, label=f"vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, p={p}")

# plt.title(f"(expected_vy-vy) for {runs} runs")
# plt.legend()
# plt.xlabel("gen")
# plt.ylabel("difference")
# plt.savefig(f"outputs/difference_vy_from_chr1_to_chr23_start0_end50_runs{runs}")





