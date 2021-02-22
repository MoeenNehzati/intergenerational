args = commandArgs(trailingOnly = T)
simdir = args[1]
print(simdir)
ped = read.table(paste(simdir,'_pedigree.txt',sep=''),header=T,stringsAsFactors=F)

ped = ped[!ped[,3]==0 & !ped[,4]==0,]

ped = ped[order(ped[,1]),]

fams = unique(ped[,1])
nfam = length(fams)

kin_out = data.frame(FID=rep(NA,5*nfam),ID1 = rep(NA,5*nfam), ID2 = rep(NA,5*nfam),
                     N_SNP =0,Z0=0,HetHet=0,IBS0=0,HetConc=0,HomIBS0=0,Kinship=0,
                     IBD1Seg=0,IBD2Seg=0,PropIBD=0,InfType=rep(NA,5*nfam),Error=0)

for (i in 1:length(fams)){
  ped_i = ped[ped[,1]==fams[i],]
  kin_out[(5*(i-1)+1):(5*(i)),'FID'] = fams[i]
  kin_out[(5*(i-1)+1):(5*(i-1)+3),'ID1'] = ped_i[1,2]
  kin_out[(5*(i-1)+1):(5*(i-1)+3),'ID2'] = c(ped_i[1,3],ped_i[1,4],ped_i[2,2])
  kin_out[(5*(i-1)+1):(5*(i-1)+3),'InfType'] = c('PO','PO','FS')
  kin_out[(5*(i-1)+4):(5*(i-1)+5),'ID1'] = ped_i[2,2]
  kin_out[(5*(i-1)+4):(5*(i-1)+5),'ID2'] = c(ped_i[1,3],ped_i[1,4])
  kin_out[(5*(i-1)+4):(5*(i-1)+5),'InfType'] = c('PO','PO')
}

gts_ids = read.table(paste(simdir,'.fam',sep=''),stringsAsFactors=F)

kin_in_gts = kin_out$ID1%in%gts_ids[,2] & kin_out$ID2%in%gts_ids[,2]

kin_out = kin_out[kin_in_gts,]

write.table(kin_out,paste(simdir,'.king.kin0',sep=''),sep='\t',quote=F,row.names=F)

individuals = ped[,2]
fathers = unique(ped[,3])
mothers = unique(ped[,4])
ped = read.table(paste(simdir,'_pedigree.txt',sep=''),header=T,stringsAsFactors=F)
agesex = data.frame(IID=c(individuals,fathers,mothers),sex=NA,age=NA)
agesex$FID = ped[match(agesex$IID,ped[,2]),1]
agesex[agesex$IID%in%individuals,'age'] = 40
agesex[agesex$IID%in%individuals,'sex'] = 'M'
agesex[agesex$IID%in%fathers,'age'] = 70
agesex[agesex$IID%in%fathers,'sex'] = 'M'
agesex[agesex$IID%in%mothers,'age'] = 70
agesex[agesex$IID%in%mothers,'sex'] = 'F'

agesex = agesex[,c(4,1:3)]
agesex = agesex[agesex$IID%in%gts_ids[,2],]
write.table(agesex,paste(simdir,'_agesex.txt',sep=''),quote=F,row.names=F)

