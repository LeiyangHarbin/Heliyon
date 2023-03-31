
###############################
rm(list=ls())
library(GSVA)
library(dorothea)
library(bcellViper)
library(dplyr)
library(viper)
library(GEOquery)
setwd("D:\\OneDrive - hrbmu.edu.cn\\yanglei004\\Deeplearning\\OV-drug-TF\\data\\")
load("OVDataSet.rda")
setwd("D:\\OneDrive - hrbmu.edu.cn\\yanglei004\\OV-dorothea\\TF activity\\")

data(dorothea_hs,package="dorothea")
regulons = dorothea_hs %>%
  filter(confidence %in% c("A","B","C","D","E"))

#save(regulons,file = "regulons.rda")
i=1
OV_TF_activities<-list()
for(i in 1:length(OVDataSet)){
  dset=OVDataSet[[i]]
  tf_activities<-run_viper(dset,regulons,
                           options=list(method="scale",minsize = 4,
                                        eset.filter=FALSE, cores=1,verbose=FALSE))
  OV_TF_activities[[i]]<-tf_activities
  names(OV_TF_activities)[i]<-names(OVDataSet)[i]
}


save(OV_TF_activities,file="OV_TF_activities.rda")

