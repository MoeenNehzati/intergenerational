from pop_sim_recom import simulate, prepare_data
import numpy as np
import h5py
import itertools
import multiprocessing
import argparse
import os
from utils import simulate_generations
import logging
logging.getLogger().setLevel(logging.INFO)

def use_derivations(phased_gts, map_list, chrom_lengths, length, a, b, p, ve, vg=None):
    if vg is None:
        vg0 = np.cov(np.sum(phased_gts, 2))
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

def f(cov, length, causal_genes, phased_gts, map_list, chrom_lengths, p, ve, mem, threads, run, bim, prefix, name, force_p=True, add_derivations=True):
    np.random.seed()
    number_of_the_genes = phased_gts.shape[0]
    pop_size = phased_gts.shape[1]
    address = f"{prefix}{name}.hdf5"
    if os.path.isfile(address):
        logging.info(f"{address} exists")
        return None, None
    else:
        logging.info(f"{address} simulating...")
    logging.info("create A and B ...")
    ab = np.random.multivariate_normal([0,0], cov, number_of_the_genes)
    print("original cov", cov)
    corr0 = np.corrcoef(ab.T)[0,1]
    a = ab[:,[0]].T
    b = ab[:,[1]].T
    causal_loci = np.random.choice(number_of_the_genes, causal_genes, False)    
    causal_mask = np.zeros(number_of_the_genes).astype(bool)
    causal_mask[causal_loci] = True
    a[0, ~causal_mask] = 0
    b[0, ~causal_mask] = 0
    temp_var = np.std(a@np.sum(phased_gts, axis=2))
    a = a/temp_var#*np.sqrt(ve)
    b = b/temp_var#*np.sqrt(ve)
    if cov[1][1] == 0:
        ve = 1
    else:

        corr_defined = cov[0][1]/np.sqrt(cov[0][0]*cov[1][1])
        initial_direct_indirect_cov = np.cov(a@np.sum(phased_gts, axis=2), b@np.sum(phased_gts, axis=2))
        initial_v_indirect = initial_direct_indirect_cov[1,1]
        initial_cov = initial_direct_indirect_cov[1,0]
        multiplier = (-initial_cov+np.sqrt(initial_cov**2+initial_v_indirect*ve))/initial_v_indirect
        b = b*multiplier
        initial_direct_indirect_cov = np.cov(a@np.sum(phased_gts, axis=2), b@np.sum(phased_gts, axis=2))
        snp_effect = initial_direct_indirect_cov[0,0] + 2*initial_direct_indirect_cov[0,1]+initial_direct_indirect_cov[1,1]/2
        ve = 2*snp_effect-2
    logging.info("A and B created")
    result = simulate(length, causal_mask, phased_gts, map_list, chrom_lengths, a, b, p, ve, mem, threads, force_p=True)
    logging.info("simulation done")
    result["causal_genes"] = np.array(causal_genes)
    result["causal_mask"] = causal_mask
    result["ab_cov"] = np.array(cov)
    result["p"] = p
    result["length"] = length
    result["ve"] = ve
    result["start"] = start
    result["end"] = end
    result["from_chr"] = from_chr
    result["to_chr"] = to_chr
    result["runs"] = runs
    result["run"] = run
    result["causal_genes"] = causal_genes    
    result["a"] = a
    result["b"] = b
    result["name"] = bytes(name, "ascii")
    result["bim_values"] = bim.values
    result["bim_columns"] = bim.columns.values.astype("S")
    if add_derivations:
        logging.info("adding expected ...")
        expected_vg, expected_vy, expected_m, expected_h2, expected_sib_cov = use_derivations(phased_gts, map_list, chrom_lengths, length, a, b, p, ve)
        logging.info("derivation done")
        result["expected_vg"] = expected_vg
        result["expected_vy"] = expected_vy
        # result["expected_m"] = expected_m
        result["expected_h2"] = expected_h2
        result["expected_sib_cov"] = expected_sib_cov
        logging.info("Added expecteds to results")

    logging.info("writing results...")    
    with h5py.File(address,'w') as f:
        for key, val in result.items():
            logging.info(f"writing key {key}")
            if type(val) == type(np.zeros(10)) or type(val) == type([]):
                print("*************** val is", key)
                #TODO fix this mess
                if type(val) == type(np.zeros(10)):
                    shape = val.shape
                if type(val) == type([]):
                    #TODO fix this
                    shape = (len(val), *val[0].shape)
                print("shape", shape)
                if (key=="males") or (key=="females"):
                    print("*************** int8", key)
                    f.create_dataset(key, shape, chunks = True, compression = "lzf", data = val, dtype=np.int8)
                else:
                    f.create_dataset(key, shape, chunks = True, compression = "lzf", data = val)
            else:
                if val is None:
                    val = np.nan
                f[key] = np.array(val)
    logging.info("writing results into {address} done")
    return name, result
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-start', type=int, help='start for each chromosome', default = 0)
    parser.add_argument('-end', type=int, help='end for each chromosome', default = None)
    parser.add_argument('-fromchr', type=int, help='starting chromosome')
    parser.add_argument('-tochr', type=int, help='until chromosome')
    parser.add_argument('-processes', type=int, help='number of processes', default = 1)
    parser.add_argument('-runs', type=int, help='number of simulations per case', default = 1)
    parser.add_argument('-length', type=int, help='simulation generations', default = 20)
    parser.add_argument('-causal', type=int, help='number of causal genes, if none all are going to be causal', default = None)
    parser.add_argument('-threads', type=int, help='parallel threads for mating', default = 10)
    parser.add_argument('-pop_size', type=int, help='parallel threads for mating', default = None)
    parser.add_argument('-d', help='direct effects', action='store_true')
    parser.add_argument('-n', help='direct and indirect but not correlated', action='store_true')
    parser.add_argument('-i', help='direct and indirect and imperfectly correlated', action='store_true')
    parser.add_argument('-p', help='direct and indirect and perfectly correlated', action='store_true')
    parser.add_argument('-a', help='all cases of direct indirect', action='store_true')
    parser.add_argument('-assort', nargs='+', help='different values of spousal phenotypic correlation', type=float, default = [0., 0.5])
    parser.add_argument('-bgen_address', help='Address of the bgen files. ~ is the wildcard for chromosome number', type=str, default = "/disk/genetics4/ukb/alextisyoung/haplotypes/sim_genotypes/QCed/chr_~")
    parser.add_argument('-add_derivations', help='Compute derivations and output them', action='store_true')
    parser.add_argument('-outprefix', help='prefix for outputs', type=str, default="")
    args=parser.parse_args()
    start = args.start
    end = args.end
    from_chr = args.fromchr
    to_chr = args.tochr
    processes = args.processes
    runs = args.runs
    length = args.length
    causal_genes = args.causal
    threads = args.threads
    pop_size = args.pop_size
    outprefix = args.outprefix
    bgen_address = args.bgen_address
    add_derivations = args.add_derivations
    ps = [a+0.000001 for a in args.assort]

    only_direct_covs = [[1, 0], [0, 0]] #only direct
    prefect_cor_covs = [[1, 0.5], [0.5, 0.25]] #perfectly correlated parental and direct
    imprefect_cor_covs = [[1, 0.25], [0.25, 0.25]] # somewhat correlated
    no_cor_covs = [[1, 0], [0, 0.25]] # somewhat correlated
    covs = []
    if args.a:
        args.d = True
        args.n = True
        args.i = True
        args.p = True
    if args.d:
        covs.append(only_direct_covs)
    if args.n:
        covs.append(no_cor_covs)
    if args.i:
        covs.append(imprefect_cor_covs)
    if args.p:
        covs.append(prefect_cor_covs)
    if len(covs) == 0:
        raise Exception("select some sort of direct/indirect effects")
    np.random.seed(11523)
    logging.info("starting prepare_data")
    phased_gts, map_list, chrom_lengths, bim = prepare_data(range(from_chr, to_chr), bgen_address, start = start, end = end)
    if not pop_size is None:
        phased_gts = phased_gts[:,:pop_size,:]
        logging.info("select first pop_size from data")
    logging.info("prepare_data done")
    number_of_the_genes = phased_gts.shape[0]
    pop_size = phased_gts.shape[1]
    fs = np.mean(np.sum(phased_gts, axis=2), axis=1)/2
    logging.info("compute fs")
    if causal_genes is None:
        causal_genes = number_of_the_genes
    ve = 1.
    mem = 2

    all_results = {}
    results = []
    class NoDaemonProcess(multiprocessing.Process):
        # make 'daemon' attribute always return False
        def _get_daemon(self):
            return False
        def _set_daemon(self, value):
            pass
        daemon = property(_get_daemon, _set_daemon)
    class MyPool(multiprocessing.pool.Pool):
        def Process(self, *args, **kwds):
            proc = super(MyPool, self).Process(*args, **kwds)
            proc.__class__ = NoDaemonProcess
            return proc

    if processes>1:
        with MyPool(processes) as pool:
            args = [(cov, length, causal_genes, phased_gts, map_list, chrom_lengths, p, ve, mem, threads, run, bim, outprefix, f"from_chr{from_chr}_to_chr{to_chr}_popsize{pop_size}_start{start}_end{end}_run{run}_p{round(p, 2)}_ab_corr{round(cov[0][1]/np.sqrt(cov[1][1]+0.000001), 2)}_vb{round(cov[1][1], 2)}_length{length}".replace(".","-"), True, add_derivations) for cov, p, run in itertools.product(covs, ps, range(runs))]
            pool.starmap(f, args)
    else:
        args = [(cov, length, causal_genes, phased_gts, map_list, chrom_lengths, p, ve, mem, threads, run, bim, outprefix, f"from_chr{from_chr}_to_chr{to_chr}_popsize{pop_size}_start{start}_end{end}_run{run}_p{round(p, 2)}_ab_corr{round(cov[0][1]/np.sqrt(cov[1][1]+0.000001), 2)}_vb{round(cov[1][1], 2)}_length{length}".replace(".","-"), True, add_derivations) for cov, p, run in itertools.product(covs, ps, range(runs))]
        for arg in args:
            f(*arg)





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
#             p = np.array(hf["p"]).item()
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             y = vy[2:]/vy[2]
#             ys[i, :] = y
#     y = np.nanmean(ys, 0)
#     plt.plot(x,y, label=f"vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, r={round(p,2)}")


# plt.title(f"Vy (normalized on the first gen) for {runs} runs")
# plt.legend(fontsize='small', loc = "lower center", ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.35)) 
# plt.xlabel("gen")
# plt.ylabel("normalized vy")
# plt.savefig(f"outputs/normalized_vy_from_chr1_to_chr23_start0_end50_runs{runs}", bbox_inches='tight')



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
#             p = np.array(hf["p"]).item()
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             y = h2[2:]/h2[2]
#             ys[i, :] = y
#     y = np.nanmean(ys, 0)
#     plt.plot(x,y, label=f"vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, r={round(p,2)}")


# plt.title(f"h2(normalized on the first gen) for {runs} runs")
# plt.legend(fontsize='small', loc = "lower center", ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.35)) 
# plt.xlabel("gen")
# plt.ylabel("normalized h2")
# plt.savefig(f"outputs/normalized_h2_from_chr1_to_chr23_start0_end50_runs{runs}", bbox_inches='tight')


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
#             p = np.array(hf["p"]).item()
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             y = h2[2:]
#             ys[i, :] = y
#     y = np.nanmean(ys, 0)
#     plt.plot(x,y, label=f"vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, r={round(p,2)}")


# plt.title(f"h2 for {runs} runs")
# plt.legend(fontsize='small', loc = "lower center", ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.35)) 
# plt.xlabel("gen")
# plt.ylabel("h2")
# plt.savefig(f"outputs/h2_from_chr1_to_chr23_start0_end50_runs{runs}", bbox_inches='tight')


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
#             p = np.array(hf["p"]).item()
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             y = vy[2:]
#             ys[i, :] = y
#     y = np.nanmean(ys, 0)
#     plt.plot(x,y, label=f"vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, r={round(p,2)}")


# plt.title(f"Vy for {runs} runs")
# plt.legend(fontsize='small', loc = "lower center", ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.35)) 
# plt.xlabel("gen")
# plt.ylabel("Vy")
# plt.savefig(f"outputs/vy_from_chr1_to_chr23_start0_end50_runs{runs}", bbox_inches='tight')


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
#             p = np.array(hf["p"]).item()
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             sim[i,:] = vy[2:]
#             expected[i,:] = vy_expected[2:]
#     sim_avg = np.nanmean(sim, 0)
#     expected_avg = np.nanmean(expected, 0)
#     plt.plot(x, sim_avg, label=f"simulation vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}")
#     plt.plot(x, expected_avg, linestyle="-.", label=f" expected vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}")

# plt.title(f"Average Vy from simulation and derivations for {runs} runs with r=0.5")
# plt.legend(fontsize='small', loc = "lower center", ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.35)) 
# plt.xlabel("gen")
# plt.ylabel("Vy")
# plt.show()
# plt.savefig(f"outputs/vy_expected_derivation_from_chr1_to_chr23_start0_end50_runs{runs}", bbox_inches='tight')


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
#             h_expected = np.array(hf["expected_h2"])            
#             ve = np.array(hf["ve"])
#             p = np.array(hf["p"]).item()
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-2)
#             sim[i,:] = h[2:]
#             expected[i,:] = h_expected[2:]
#     sim_avg = np.nanmean(sim, 0)
#     expected_avg = np.nanmean(expected, 0)
#     plt.plot(x, sim_avg, label=f"simulation vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}")
#     plt.plot(x, expected_avg, linestyle="-.", label=f" expected vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}")

# plt.title(f"Average h2 from simulation and derivations for {runs} runs with r=0.5")
# plt.legend(fontsize='small', loc = "lower center", ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.35)) 
# plt.xlabel("gen")
# plt.ylabel("Vy")
# plt.savefig(f"outputs/h2_expected_derivation_from_chr1_to_chr23_start0_end50_runs{runs}", bbox_inches='tight')




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
#             p = np.array(hf["p"]).item()
#             ab_cov = np.array(hf["ab_cov"])
#             x = range(len(vy)-3)
#             ys[i, :] = vy[2:]#(vy[3:]-vy[2:-1])/vy[2:-1]
#     y = np.nanmean(ys, 0)
#     y = y[1:]-y[:-1]
#     plt.plot(x,y, label=f"vb={ab_cov[1,1]}, corr(a,b)={round(ab_cov[0,1]/(np.sqrt(ab_cov[1,1])+ 0.00001),2)}, r={round(p,2)}")


# plt.title(f"Convergence speed for {runs} runs")
# plt.legend(fontsize='small')
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





