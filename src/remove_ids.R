args = commandArgs(trailingOnly = T)

bfile = args[1]

fam = read.table(paste(bfile,'fam',sep='.'),stringsAsFactors =F)

gen_1 = fam[fam[,3]!=0,]
gen_0 = fam[fam[,3]==0,]

gen_1 = gen_1[order(gen_1[,1]),]

bpg_remove_fams = 1:10000
opg_remove_fams = 10001:20000

remove = c(gen_1[gen_1[,1]%in%bpg_remove_fams,3],
           gen_1[gen_1[,1]%in%bpg_remove_fams,4],
           gen_1[gen_1[,1]%in%opg_remove_fams,3])

opg_sibs = gen_1[gen_1[,1]%in%opg_remove_fams,2]

remove = c(remove,opg_sibs[seq(1,length(opg_sibs)/2,2)])

remove = unique(remove)

write.table(fam[match(remove,fam[,2]),1:2],paste(bfile,'remove.txt',sep='_'),
            quote=F,row.names=F,col.names=F)

write.table(data.frame(FID=fam[,1],IID=fam[,2],FATHER_ID=fam[,3],MOTHER_ID=fam[,4]),paste(bfile,'pedigree.txt',sep='_'),quote=F,row.names=F)
write.table(data.frame(FID=gen_1[,1],IID=gen_1[,2],phenotype=gen_1[,6]),paste(bfile,'phenotype.txt',sep='_'),quote=F,row.names=F)
