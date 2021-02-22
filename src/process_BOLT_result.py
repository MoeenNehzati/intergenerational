import h5py
import numpy as np
import pandas as pd
import argparse
from scipy.stats import linregress
import statsmodels.api as sm
import itertools
from matplotlib import pyplot as plt


def check_one_file(prefix, name, bolt_pref="bolt/"):
    print(prefix)
    print(name)
    with h5py.File(prefix+name+".hdf5", "r") as hf:
        true_a = np.array(hf["a"]).flatten()
        true_b = np.array(hf["b"]).flatten()
    bolt_pd = pd.read_csv(prefix+bolt_pref+name+"_results.csv", sep="\t")
    print("this file:", prefix+bolt_pref+name+"_results.csv")
    bolt_std = bolt_pd["SE"].values
    bolt_estimate = bolt_pd["BETA"].values
    return bolt_estimate, bolt_std, true_a, true_b

def all_files(prefix="outputs/multi_runs/", runs = 20, bolt_pref="bolt/"):
    vb = ["0", "0-25"]
    ab_corr = ["0-0", "0-5", "1-0"]
    # p = ["0-0", "0-5"]
    p = ["0-0"]
    with_indirect = list(itertools.product(p, ab_corr, [vb[1]]))
    without_indirect = list(itertools.product(p, ["0-0"], [vb[0]]))
    param_sets = without_indirect+with_indirect
    regression_on_a = []
    regression_on_a_plus_b = []
    regression_on_ab = []
    for p, ab_corr, vb in param_sets:
        bolt_estimates = []
        bolt_stds = []
        true_as = []
        true_bs = []
        for run in range(runs):
            # name = f"from_chr1_to_chr23_start0_end50_run{run}_p{p}_ab_corr{ab_corr}_vb{vb}_length15"
            name = f"from_chr1_to_chr23_start0_endNone_run{run}_p{p}_ab_corr{ab_corr}_vb{vb}_length2"
            bolt_estimate, bolt_std, true_a, true_b = check_one_file(prefix, name, bolt_pref)
            bolt_estimates.append(bolt_estimate)
            bolt_stds.append(bolt_std)
            true_as.append(true_a)
            true_bs.append(true_b)
        bolt_estimates = np.hstack(bolt_estimates)
        bolt_stds = np.hstack(bolt_stds)
        true_as = np.hstack(true_as)
        true_bs = np.hstack(true_bs)
        weighted_estimates = bolt_estimates/bolt_stds
        weighted_true_as = true_as/bolt_stds
        weighted_true_bs = true_bs/bolt_stds
        weighted_estimates[np.isnan(weighted_estimates)] = 0
        weighted_true_as[np.isnan(weighted_true_as)] = 0
        weighted_true_bs[np.isnan(weighted_true_bs)] = 0
        print(np.sum(np.isnan(weighted_estimates-weighted_true_as)))
        print(np.sum(np.isinf(weighted_estimates-weighted_true_as)))
        print(np.sum(np.isnan(np.vstack(( weighted_true_as, np.ones(weighted_true_as.shape) )).T)))
        print(np.sum(np.isinf(np.vstack(( weighted_true_as, np.ones(weighted_true_as.shape) )).T)))

        ols = sm.OLS(weighted_estimates, np.vstack(( weighted_true_as, np.ones(weighted_true_as.shape) )).T )
        ols_result = ols.fit()
        params = ols_result.params
        p_values = ols_result.pvalues
        std_err = ols_result.bse
        regression_on_a.append([p, ab_corr, vb, params[0], p_values[0], std_err[0], params[1], p_values[1], std_err[1]])

        ols = sm.OLS(weighted_estimates, np.vstack(( weighted_true_as+weighted_true_bs, np.ones(weighted_true_as.shape) )).T )
        ols_result = ols.fit()
        params = ols_result.params
        p_values = ols_result.pvalues
        std_err = ols_result.bse
        regression_on_a_plus_b.append([p, ab_corr, vb, params[0], p_values[0], std_err[0], params[1], p_values[1], std_err[1]])

        ols = sm.OLS(weighted_estimates, np.vstack(( weighted_true_as, weighted_true_bs, np.ones(weighted_true_bs.shape) )).T )
        ols_result = ols.fit()
        params = ols_result.params
        p_values = ols_result.pvalues
        std_err = ols_result.bse
        regression_on_ab.append([p, ab_corr, vb, params[0], p_values[0], std_err[0], params[1], p_values[1], std_err[1], params[2], p_values[2], std_err[2]])

    pd_a = pd.DataFrame(regression_on_a, columns=["p", "ab_corr", "vb", "a", "a_p_value", "a_std_err", "intercept", "intercept_p_value", "intercept_std_err"])
    pd_a_plus_b = pd.DataFrame(regression_on_a_plus_b, columns=["p", "ab_corr", "vb", "a+b", "a+b_p_value", "a+b_std_err", "intercept", "intercept_p_value", "intercept_std_err"])
    pd_ab = pd.DataFrame(regression_on_ab, columns=["p", "ab_corr", "vb", "a", "a_p_value", "a_std_err", "b", "b_p_value", "b_std_err", "intercept", "intercept_p_value", "intercept_std_err"])
    return pd_a, pd_a_plus_b, pd_ab

def process_single_ldsc(run=0):
    h2_pref = "Total Observed scale h2: "
    vb = ["0", "0-25"]
    ab_corr = ["0-0", "0-5", "1-0"]
    # p = ["0-0", "0-5"]
    p = ["0-0"]
    with_indirect = list(itertools.product(p, ab_corr, [vb[1]]))
    without_indirect = list(itertools.product(p, ["0-0"], [vb[0]]))
    param_sets = without_indirect+with_indirect
    results_pd = []
    for p, ab_corr, vb in param_sets:
        diffs = []
        estimates  = []
        name = f"outputs/multi_runs/ldsc_regression/from_chr1_to_chr23_start0_endNone_run{run}_p{p}_ab_corr{ab_corr}_vb{vb}_length2_results.log"
        print(name)
        with open(name, "r") as f:
            log = f.read()
        h2_start = log.find(h2_pref)+len(h2_pref)
        length = log[h2_start:].find("(")
        std_end = log[h2_start:].find(")")
        ldsc_h2 = float(log[h2_start:h2_start+length]) 
        ldsc_std = float(log[h2_start+length+2:h2_start+std_end])           
        # with h5py.File(f"outputs/multi_runs/from_chr1_to_chr23_start0_end50_run{run}_p{p}_ab_corr{ab_corr}_vb{vb}_length15.hdf5", "r") as hf:
        with h5py.File(f"outputs/multi_runs/from_chr1_to_chr23_start0_endNone_run{run}_p{p}_ab_corr{ab_corr}_vb{vb}_length2.hdf5", "r") as hf:
            true_h2 = np.array(hf["heritablity"])
        results_pd.append([p, ab_corr, vb, ldsc_h2, ldsc_std, true_h2[0], true_h2[1]])
    results_pd = pd.DataFrame(results_pd, columns = ["p", "ab_corr", "vb", "ldsc_h2", "ldsc_SE", "initial h2",  "equilibrium h2"])
    return results_pd


def process_ldsc_regression(runs=20):
    h2_pref = "Total Observed scale h2: "
    vb = ["0", "0-25"]
    ab_corr = ["0-0", "0-5", "1-0"]
    # p = ["0-0", "0-5"]
    p = ["0-0"]
    with_indirect = list(itertools.product(p, ab_corr, [vb[1]]))
    without_indirect = list(itertools.product(p, ["0-0"], [vb[0]]))
    param_sets = without_indirect+with_indirect
    results_pd = []
    for p, ab_corr, vb in param_sets:
        diffs = []
        estimates  = []
        for run in range(runs):
            # name = f"outputs/multi_runs/ldsc_regression/from_chr1_to_chr23_start0_end50_run{run}_p{p}_ab_corr{ab_corr}_vb{vb}_length15_results.log"
            name = f"outputs/multi_runs/ldsc_regression/from_chr1_to_chr23_start0_endNone_run{run}_p{p}_ab_corr{ab_corr}_vb{vb}_length2_results.log"
            print(name)
            with open(name, "r") as f:
                log = f.read()
            h2_start = log.find(h2_pref)+len(h2_pref)
            length = log[h2_start:].find("(")
            ldsc_h2 = float(log[h2_start:h2_start+length])            
            # with h5py.File(f"outputs/multi_runs/from_chr1_to_chr23_start0_end50_run{run}_p{p}_ab_corr{ab_corr}_vb{vb}_length15.hdf5", "r") as hf:
            with h5py.File(f"outputs/multi_runs/from_chr1_to_chr23_start0_endNone_run{run}_p{p}_ab_corr{ab_corr}_vb{vb}_length2.hdf5", "r") as hf:
                true_h2 = np.array(hf["heritablity"])
            diff = ldsc_h2-true_h2
            diffs.append(diff)
            estimates.append(ldsc_h2)
        diffs = np.array(diffs)
        estimates = np.array(estimates)
        diffs = diffs.mean(0)
        estimates = estimates.mean()
        results_pd.append([p, ab_corr, vb, estimates, diffs[0], diffs[-1]])
    results_pd = pd.DataFrame(results_pd, columns = ["p", "ab_corr", "vb", "ldsc_h2", "average ldsc_h2-initial_h2", "average ldsc_h2-equilibrium_h2"])
    return results_pd

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("address", type=str, help='hdf5 address')
    # args=parser.parse_args()
    # address = args.address
    # print("=====================================")
    # print("=====================================")
    # print("=====================================")
    # print("=====================================")
    # print("=====================================")
    # print(address)
    # print(check_one_file(address))


    # pd_a, pd_a_plus_b, pd_ab=all_files()
    # pd_ldsc = process_ldsc_regression()
    pd_a, pd_a_plus_b, pd_ab=all_files(runs=1)
    pd_ldsc = process_ldsc_regression(1)
    graphs = [
        [pd_a, "pd_a", "regressing bolt_estimate-true_a on true_a \n everything is weighted by 1/estimate_std"],
        [pd_a_plus_b, "pd_a_plus_b", "regressing bolt_estimate-(true_a+true_b) on true_a+true_b \n everything is weighted by 1/estimate_std"],
        [pd_ab, "pd_ab", "regressing bolt_estimate on true_a and true_b jointly \n everything is weighted by 1/estimate_std"],
        [pd_ldsc, "pd_ldsc", "average lscd_h2-h2"],
    ]
    for g in graphs:
        plt.clf()
        plt.axis("off")
        tab = plt.table(cellText=g[0].round(3).values, colLabels=g[0].columns, loc='center', cellLoc='center')
        tab.auto_set_column_width(col=list(range(len(g[0].columns)))) # Provide integer list of c
        plt.title(g[2])
        plt.show()
        plt.savefig(g[1], bbox_inches='tight', dpi=300, pad_inches = 0)

# python -c "from process_BOLT_result import all_files;pd_a, pd_a_plus_b, pd_ab=all_files();pd_a = pd_a.round(3);pd_a_plus_b = pd_a_plus_b.round(3);pd_a_plus_ab = pd_a_plus_ab.round(3);pd_a.to_html('pd_a.html');pd_a_plus_b.to_html('pd_a_plus_b.html');pd_ab.to_html('pd_ab.html');print('pd_a');print(pd_a);print('pd_a_plus_b');print(pd_a_plus_b);print('pd_ab');print(pd_ab)"


# from process_BOLT_result import all_files
# pd_a, pd_a_plus_b, pd_a_plus_ab=all_files()
# pd_a = pd_a.round(3)
# pd_a_plus_b = pd_a_plus_b.round(3)
# pd_a_plus_ab = pd_a_plus_ab.round(3)
# pd_a.to_html('pd_a.html')
# pd_a_plus_b.to_html('pd_a_plus_b.html')
# pd_a_plus_ab.to_html('pd_a_plus_ab.html')
# print('pd_a')
# print(pd_a)
# print('pd_a_plus_b')
# print(pd_a_plus_b)
# print('pd_a_plus_ab')
# print(pd_a_plus_ab)