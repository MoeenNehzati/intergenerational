import pandas as pd
def convert(address):
    bolt_df = pd.read_csv(address, sep="\t")[["SNP",	"ALLELE1",	"ALLELE0",	"BETA", "SE"]]
    bolt_df["Z"] = bolt_df["BETA"]/bolt_df["SE"]
    bolt_df = bolt_df.rename(columns={
        "A1":"ALLELE0",
        "A2":"ALLELE1",
    })
    bolt_df["N"] = 98000
    ldsc_df = bolt_df[["SNP",	"ALLELE1",	"ALLELE0",	"Z",	"N"]]    
    name = address.split("/")[-1]
    pref = address[:-(len(name)+5)]
    ldsc_df.to_csv(pref + "sumstats/" + name[:-4] + ".sumstats", sep="\t")

# for x in `ls outputs/multi_runs/*.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/${x}_generation1.bed ]; then
# python -c "from pop_sim_recom import write_bed_seg;write_bed_seg('outputs/multi_runs/${x}.hdf5')"
# else
# echo passed;
# fi;
# done


# for x in `ls outputs/multi_runs/*.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/bolt/${x}_results.csv ]; then
# /var/genetics/pub/software/bolt/BOLT-LMM_v2.3/bolt --lmm \
# --LDscoresFile=/disk/genetics/pub/software/bolt/BOLT-LMM_v2.3/tables/LDSCORE.1000G_EUR.tab.gz \
# --bfile outputs/multi_runs/${x}_generation1 \
# --phenoUseFam \
# --statsFile outputs/multi_runs/bolt/${x}_results.csv \
# --numThreads 20 \
# --noMapCheck
# fi;
# done


# for x in `ls outputs/multi_runs/bolt/*.csv | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/sumstats/${x}.sumstats ]; then
# python -c "from convert_bolt_ldsc import convert; convert('outputs/multi_runs/bolt/${x}.csv')";
# fi;
# done


# for x in `ls outputs/multi_runs/*.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/bolt/${x}_results.csv ]; then
# python2 ../ldsc/ldsc.py \
# 	--bfile outputs/multi_runs/${x}_generation1 \
# 	--l2 \
# 	--ld-wind-cm 1 \
# 	--out outputs/multi_runs/LDScore/${x}_results
# fi;
# done


# for x in `ls outputs/multi_runs/sumstats/*.sumstats | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/ldsc_regression/${x}.log ]; then
# python2 ../ldsc/ldsc.py \
# --h2 outputs/multi_runs/sumstats/${x}.sumstats \
# --ref-ld outputs/multi_runs/LDScore/${x} \
# --w-ld outputs/multi_runs/LDScore/${x} \
# --out outputs/multi_runs/ldsc_regression/${x}
# fi;
# done
# less outputs/multi_runs/ldsc_regression/from_chr1_to_chr23_start0_end50_run61_p0-05_ab_corr1-0_vb0-25_length15.log










# for x in `ls outputs/multi_runs/*end50_run0*.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`; do\
#     for chr in {1..22}; do \
#         if [ ! -f outputs/multi_runs/by_chrom/${x}_generation1_chr${chr}.bed ]; then
#             plink --bfile outputs/multi_runs/${x}_generation1 --chr $chr --make-bed --out outputs/multi_runs/by_chrom/${x}_generation1_chr${chr}; \
#         fi
#     done
# done

# for x in `ls outputs/multi_runs/*end50_run0*.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`; do\
#     for chr in {1..22}; do \
#         if [ ! -f outputs/multi_runs/by_chrom/LDScore/${x}_results_chr${chr}.l2.ldscore.gz ]; then
#             python ../ldsc/ldsc.py --bfile outputs/multi_runs/by_chrom/${x}_generation1_chr${chr} --l2 --out outputs/multi_runs/by_chrom/LDScore/${x}_results_chr${chr} --yes-really --ld-wind-snp 50
#         fi
#     done
# done



# for x in `ls outputs/multi_runs/*end50_run0*.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/by_chrom/LDScore/${x}.afreq ]; then
#     plink -bfile outputs/multi_runs/${x}_generation1 --freq --nonfounders --out outputs/multi_runs/by_chrom/LDScore/${x}
# else
#     echo passed;
# fi;
# if [ ! -f outputs/multi_runs/by_chr/LDScore/${x}_results.l2.ldscore.gz ]; then
# python -c "import pandas as pd
# lds = [pd.read_csv('outputs/multi_runs/by_chrom/LDScore/${x}_results_chr'+str(ld)+'.l2.ldscore.gz', sep='\t') for ld in range(1,23)]
# ld = pd.concat(lds)
# maf = pd.read_csv('outputs/multi_runs/by_chrom/LDScore/${x}.afreq', sep='\t')
# print("ld", ld)
# ld['MAF'] = maf['ALT_FREQS']
# ld = ld.rename(columns={'L2':'LDSCORE'})
# ld = ld[['SNP', 'CHR', 'BP', 'MAF', 'LDSCORE']]
# ld.to_csv('outputs/multi_runs/by_chrom/LDScore/${x}_results.l2.ldscore.gz', sep='\t', index = False)"
# fi;
# done



# for x in `ls outputs/multi_runs/*end50_run0*.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/LDScore_bolt/${x}_results.csv ]; then
# /var/genetics/pub/software/bolt/BOLT-LMM_v2.3/bolt --lmm \
# --LDscoresFile=outputs/multi_runs/by_chrom/LDScore/${x}_results.l2.ldscore.gz \
# --bfile outputs/multi_runs/${x}_generation1 \
# --phenoUseFam \
# --statsFile outputs/multi_runs/LDScore_bolt/${x}_results.csv \
# --numThreads 20 \
# --noMapCheck
# fi;
# done












#===================================================================
#===================================================================
#===================================================================
#==========================small scale==============================
#===================================================================
#===================================================================
#===================================================================
#===================================================================

# for x in `ls outputs/multi_runs/*_run{,1}[0-9]_*_length15.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/${x}_generation1.bed ]; then
# python -c "from pop_sim_recom import write_bed_seg;write_bed_seg('outputs/multi_runs/${x}.hdf5')"
# else
# echo passed;
# fi;
# done


# for x in `ls outputs/multi_runs/*_run{,1}[0-9]_*_length15.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/bolt/${x}_results.csv ]; then
# /var/genetics/pub/software/bolt/BOLT-LMM_v2.3/bolt --lmm \
# --LDscoresFile=/disk/genetics/pub/software/bolt/BOLT-LMM_v2.3/tables/LDSCORE.1000G_EUR.tab.gz \
# --bfile outputs/multi_runs/${x}_generation1 \
# --phenoUseFam \
# --statsFile outputs/multi_runs/bolt/${x}_results.csv \
# --numThreads 5 \
# --noMapCheck
# fi;
# done


# for x in `ls outputs/multi_runs/bolt/*_run{,1}[0-9]_*.csv | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/sumstats/${x}.sumstats ]; then
# python -c "from convert_bolt_ldsc import convert;convert('outputs/multi_runs/bolt/${x}.csv')";
# fi;
# done


# for x in `ls outputs/multi_runs/*_run{,1}[0-9]_*_length15.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/LDScore/${x}_results.l2.ldscore.gz ]; then
# echo "not pass ${x}"
# python2 ../ldsc/ldsc.py \
# 	--bfile outputs/multi_runs/${x}_generation1 \
# 	--l2 \
# 	--ld-wind-cm 1 \
# 	--out outputs/multi_runs/LDScore/${x}_results
# else
# echo "pass ${x}";
# fi;
# done


# for x in `ls outputs/multi_runs/sumstats/*run{,1}[0-9]_*.sumstats | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/ldsc_regression/${x}.log ]; then
# python2 ../ldsc/ldsc.py \
# --h2 outputs/multi_runs/sumstats/${x}.sumstats \
# --ref-ld outputs/multi_runs/LDScore/${x} \
# --w-ld outputs/multi_runs/LDScore/${x} \
# --out outputs/multi_runs/ldsc_regression/${x}
# fi;
# done
# less outputs/multi_runs/ldsc_regression/from_chr1_to_chr23_start0_end50_run61_p0-05_ab_corr1-0_vb0-25_length15.log






#===================================================================
#===================================================================
#===================================================================
#==========================large scale==============================
#===================================================================
#===================================================================
#===================================================================
#===================================================================_endNone

# for x in `ls outputs/multi_runs/*_endNone_run{,1}[0-9]_*_length2.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/${x}_generation1.bed ]; then
# python -c "from pop_sim_recom import write_bed_seg;write_bed_seg('outputs/multi_runs/${x}.hdf5')"
# else
# echo passed;
# fi;
# done


# for x in `ls outputs/multi_runs/*_endNone_run{,1}[0-9]_*_length2.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/bolt/${x}_results.csv ]; then
# /var/genetics/pub/software/bolt/BOLT-LMM_v2.3/bolt --lmm \
# --LDscoresFile=/disk/genetics/pub/software/bolt/BOLT-LMM_v2.3/tables/LDSCORE.1000G_EUR.tab.gz \
# --bfile outputs/multi_runs/${x}_generation1 \
# --phenoUseFam \
# --statsFile outputs/multi_runs/bolt/${x}_results.csv \
# --numThreads 5 \
# --noMapCheck
# fi;
# done


# for x in `ls outputs/multi_runs/bolt/*_endNone_run{,1}[0-9]_*.csv | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/sumstats/${x}.sumstats ]; then
# python -c "from convert_bolt_ldsc import convert;convert('outputs/multi_runs/bolt/${x}.csv')";
# fi;
# done


# for x in `ls outputs/multi_runs/*_endNone_run{,1}[0-9]_*_length2.hdf5 | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/LDScore/${x}_results.l2.ldscore.gz ]; then
# echo "not pass ${x}"
# python2 ../ldsc/ldsc.py \
# 	--bfile outputs/multi_runs/${x}_generation1 \
# 	--l2 \
# 	--ld-wind-cm 1 \
# 	--out outputs/multi_runs/LDScore/${x}_results
# else
# echo "pass ${x}";
# fi;
# done


# for x in `ls outputs/multi_runs/sumstats/*_endNone_run{,1}[0-9]_*.sumstats | sed -r "s/.+\/(.+)\..+/\1/"`;
# do if [ ! -f outputs/multi_runs/ldsc_regression/${x}.log ]; then
# python2 ../ldsc/ldsc.py \
# --h2 outputs/multi_runs/sumstats/${x}.sumstats \
# --ref-ld outputs/multi_runs/LDScore/${x} \
# --w-ld outputs/multi_runs/LDScore/${x} \
# --out outputs/multi_runs/ldsc_regression/${x}
# fi;
# done
# less outputs/multi_runs/ldsc_regression/from_chr1_to_chr23_start0_end50_run61_p0-05_ab_corr1-0_vb0-25_length15.log
