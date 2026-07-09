import sys
sys.path.insert(0, '/Users/jsmonzon/Research/SatGen/mcmc/src/')
import jsm_ancillary

DF_tags = ["DF_01", "DF_1", "DF_2", "DF_3", "DF_4", "DF_5", "DF_10"]

for tag in DF_tags:

    temp = jsm_ancillary.load_massspec_z0("../../data/zhao/DF_test/"+tag+"/", "artificial")
    temp.to_csv("../../data/zhao/DF_test/"+tag+"/artificial_all.csv")