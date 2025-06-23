"""Crude codes for plotting the 2022 data-MC comparison plots. """
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os

# This piece of codes can be used to simplify this script.
# preEE_luminosity = 7.9804 * 1000
# postEE_luminosity = 26.6717 * 1000
# with open('/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/cross_sections_for_Ntuples_v3.json', 'r') as f:
#     cross_sections = json.load(f)
# with open('/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022preEE/2022preEE_total_sum_genWeight_using_Ntuples_v3.json', 'r') as f:
#     preEE_sum_genWeight = json.load(f)
# with open('/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022postEE/2022postEE_total_sum_genWeight_using_Ntuples_v3.json', 'r') as f:
#     postEE_sum_genWeight = json.load(f)

# def get_weight(ntuple_name, ntuple):
#     ntuple_name_without_suffix = ntuple_name

#     if "_preEE" in ntuple_name:
#         ntuple_name_without_suffix = ntuple_name.replace("_preEE", "")
#         luminosity = preEE_luminosity
#         total_sum_genWeight = preEE_sum_genWeight.get(ntuple_name, 1)
#     elif "_postEE" in ntuple_name:
#         ntuple_name_without_suffix = ntuple_name.replace("_postEE", "")
#         luminosity = postEE_luminosity
#         total_sum_genWeight = postEE_sum_genWeight.get(ntuple_name, 1)
#     else:
#         luminosity = 1
#         total_sum_genWeight = 1

#     if ntuple_name_without_suffix in cross_sections:
#         xsec = cross_sections[ntuple_name_without_suffix]
#         return luminosity * xsec * (ntuple.genWeight / total_sum_genWeight)
#     else:
#         return None

# ntuples_dict = {}
# basepath = {
#     "preEE" : "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022preEE/WH_xgboost_training/training/Backgrounds",
#     "postEE" : "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022postEE/WH_xgboost_training/training/Backgrounds"
# }

# for tag, path in basepath.items():
#     for category in os.listdir(basepath[tag]):
#         category_path = os.path.join(basepath[tag], category)
#         print(f"INFO: {category_path} is loaded.")
#         for process in os.listdir(category_path):
#             ntuple_path = os.path.join(category_path, process, "nominal")
#             ntuple_name = process.replace("-", "_")
#             ntuples_dict[ntuple_name] = ak.from_parquet(ntuple_path)

# print(f"2022 preEE and postEE background Ntuples are loaded.")
# print("\n")

# ntuples_list = list(ntuples_dict.values())

preEE_basepath  = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022preEE/Ntuples_v3"
postEE_basepath = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022postEE/Ntuples_v3"

################################################################################################################################
########## Load the Ntuples of 2022 preEE and postEE data ##########
################################################################################################################################
DataC_preEE  = ak.from_parquet(f"{preEE_basepath}/Data/DataC_2022/nominal")
DataD_preEE  = ak.from_parquet(f"{preEE_basepath}/Data/DataD_2022/nominal")
print("[INFO] All 2022 preEE data Ntuples are loaded.")
# ------------------------------------------------------------------------------------------------------------------------------
DataE_postEE = ak.from_parquet(f"{postEE_basepath}/Data/DataE_2022/nominal")
DataF_postEE = ak.from_parquet(f"{postEE_basepath}/Data/DataF_2022/nominal")
DataG_postEE = ak.from_parquet(f"{postEE_basepath}/Data/DataG_2022/nominal")
print("[INFO] All 2022 postEE data Ntuples are loaded.")
################################################################################################################################
########## Load the Ntuples of 2022 preEE and postEE MC samples ##########
################################################################################################################################
########################
########## VV ##########
########################
WW_preEE  = ak.from_parquet(f"{preEE_basepath}/VV/WW_preEE/nominal")
WZ_preEE  = ak.from_parquet(f"{preEE_basepath}/VV/WZ_preEE/nominal")
ZZ_preEE  = ak.from_parquet(f"{preEE_basepath}/VV/ZZ_preEE/nominal")
print("[INFO] Ntuples of 2022 preEE VV samples are loaded.")
# ------------------------------------------------------------------------------------------------------------------------------
WW_postEE = ak.from_parquet(f"{postEE_basepath}/VV/WW_postEE/nominal")
WZ_postEE = ak.from_parquet(f"{postEE_basepath}/VV/WZ_postEE/nominal")
ZZ_postEE = ak.from_parquet(f"{postEE_basepath}/VV/ZZ_postEE/nominal")
print("[INFO] Ntuples of 2022 postEE VV samples are loaded.")
###########################################
########## Diphoton-nonresonance ##########
###########################################
Diphoton_preEE  = ak.from_parquet(f"{preEE_basepath}/Diphoton/GG-Box-3Jets_MGG-80_preEE/nominal")
print("[INFO] Ntuples of 2022 preEE diphoton samples are loaded.")
# ------------------------------------------------------------------------------------------------------------------------------
Diphoton_postEE = ak.from_parquet(f"{postEE_basepath}/Diphoton/GG-Box-3Jets_MGG-80_postEE/nominal")
print("[INFO] Ntuples of 2022 postEE diphoton samples are loaded.")
############################
########## DYJets ##########
############################
DYto2L_2Jets_MLL_50_preEE                 = ak.from_parquet(f"{preEE_basepath}/DYJets/DYto2L-2Jets_MLL-50_preEE/nominal")
DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE    = ak.from_parquet(f"{preEE_basepath}/DYJets/DYGto2LG-1Jets_MLL-50_PTG-10to50_preEE/nominal")
DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE   = ak.from_parquet(f"{preEE_basepath}/DYJets/DYGto2LG-1Jets_MLL-50_PTG-50to100_preEE/nominal")
DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE  = ak.from_parquet(f"{preEE_basepath}/DYJets/DYGto2LG-1Jets_MLL-50_PTG-100to200_preEE/nominal")
DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE  = ak.from_parquet(f"{preEE_basepath}/DYJets/DYGto2LG-1Jets_MLL-50_PTG-200to400_preEE/nominal")
DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE  = ak.from_parquet(f"{preEE_basepath}/DYJets/DYGto2LG-1Jets_MLL-50_PTG-400to600_preEE/nominal")
DYGto2LG_1Jets_MLL_50_PTG_600_preEE       = ak.from_parquet(f"{preEE_basepath}/DYJets/DYGto2LG-1Jets_MLL-50_PTG-600_preEE/nominal")
print("[INFO] Ntuples of 2022 preEE DY samples are loaded.")
# ------------------------------------------------------------------------------------------------------------------------------
DYto2L_2Jets_MLL_50_postEE                = ak.from_parquet(f"{postEE_basepath}/DYJets/DYto2L-2Jets_MLL-50_postEE/nominal")
DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE   = ak.from_parquet(f"{postEE_basepath}/DYJets/DYGto2LG-1Jets_MLL-50_PTG-10to50_postEE/nominal")
DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE  = ak.from_parquet(f"{postEE_basepath}/DYJets/DYGto2LG-1Jets_MLL-50_PTG-50to100_postEE/nominal")
DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE = ak.from_parquet(f"{postEE_basepath}/DYJets/DYGto2LG-1Jets_MLL-50_PTG-100to200_postEE/nominal")
DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE = ak.from_parquet(f"{postEE_basepath}/DYJets/DYGto2LG-1Jets_MLL-50_PTG-200to400_postEE/nominal")
DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE = ak.from_parquet(f"{postEE_basepath}/DYJets/DYGto2LG-1Jets_MLL-50_PTG-400to600_postEE/nominal")
DYGto2LG_1Jets_MLL_50_PTG_600_postEE      = ak.from_parquet(f"{postEE_basepath}/DYJets/DYGto2LG-1Jets_MLL-50_PTG-600_postEE/nominal")
print("[INFO] Ntuples of 2022 postEE DY samples are loaded.")
###########################
########## GJets ##########
###########################
GJet_PT_20_MGG_40to80_preEE  = ak.from_parquet(f"{preEE_basepath}/GJets/GJet_PT-20_DoubleEMEnriched_MGG-40to80_preEE/nominal")
GJet_PT_20to40_MGG_80_preEE  = ak.from_parquet(f"{preEE_basepath}/GJets/GJet_PT-20to40_DoubleEMEnriched_MGG-80_preEE/nominal")
GJet_PT_40_MGG_80_preEE      = ak.from_parquet(f"{preEE_basepath}/GJets/GJet_PT-40_DoubleEMEnriched_MGG-80_preEE/nominal")
print("[INFO] Ntuples of 2022 preEE GJets samples are loaded.")
# ------------------------------------------------------------------------------------------------------------------------------
GJet_PT_20_MGG_40to80_postEE = ak.from_parquet(f"{postEE_basepath}/GJets/GJet_PT-20_DoubleEMEnriched_MGG-40to80_postEE/nominal")
GJet_PT_20to40_MGG_80_postEE = ak.from_parquet(f"{postEE_basepath}/GJets/GJet_PT-20to40_DoubleEMEnriched_MGG-80_postEE/nominal")
GJet_PT_40_MGG_80_postEE     = ak.from_parquet(f"{postEE_basepath}/GJets/GJet_PT-40_DoubleEMEnriched_MGG-80_postEE/nominal") 
print("[INFO] Ntuples of 2022 postEE GJets samples are loaded.")
#########################
########## QCD ##########
#########################
QCD_PT_30to40_MGG_80toInf_preEE   = ak.from_parquet(f"{preEE_basepath}/QCD/QCD_PT-30to40_DoubleEMEnriched_MGG-80toInf_preEE/nominal")
QCD_PT_40toInf_MGG_80toInf_preEE  = ak.from_parquet(f"{preEE_basepath}/QCD/QCD_PT-40toInf_DoubleEMEnriched_MGG-80toInf_preEE/nominal")
print("[INFO] Ntuples of 2022 preEE QCD samples are loaded.")
# ------------------------------------------------------------------------------------------------------------------------------
QCD_PT_30to40_MGG_80toInf_postEE  = ak.from_parquet(f"{postEE_basepath}/QCD/QCD_PT-30to40_DoubleEMEnriched_MGG-80toInf_postEE/nominal")
QCD_PT_40toInf_MGG_80toInf_postEE = ak.from_parquet(f"{postEE_basepath}/QCD/QCD_PT-40toInf_DoubleEMEnriched_MGG-80toInf_postEE/nominal")
print("[INFO] Ntuples of 2022 postEE QCD samples are loaded.")
#########################
########## Top ##########
#########################
TGJets_preEE                  = ak.from_parquet(f"{preEE_basepath}/Top/TGJets_preEE/nominal")
TTG_1Jets_PTG_10to100_preEE   = ak.from_parquet(f"{preEE_basepath}/Top/TTG-1Jets_PTG-10to100_preEE/nominal")
TTG_1Jets_PTG_100to200_preEE  = ak.from_parquet(f"{preEE_basepath}/Top/TTG-1Jets_PTG-100to200_preEE/nominal")
TTG_1Jets_PTG_200_preEE       = ak.from_parquet(f"{preEE_basepath}/Top/TTG-1Jets_PTG-200_preEE/nominal")
TTGG_0Jets_preEE              = ak.from_parquet(f"{preEE_basepath}/Top/TTGG_0Jets_preEE/nominal")
TTto2L2Nu_2Jets_preEE         = ak.from_parquet(f"{preEE_basepath}/Top/TTto2L2Nu-2Jets_preEE/nominal")
TTto4Q_2Jets_preEE            = ak.from_parquet(f"{preEE_basepath}/Top/TTto4Q-2Jets_preEE/nominal")
TTtoLminusNu2Q_2Jets_preEE    = ak.from_parquet(f"{preEE_basepath}/Top/TTtoLminusNu2Q-2Jets_preEE/nominal")
TTtoLplusNu2Q_2Jets_preEE     = ak.from_parquet(f"{preEE_basepath}/Top/TTtoLplusNu2Q-2Jets_preEE/nominal")
print("[INFO] Ntuples of 2022 preEE Top quarks samples are loaded.")
# ------------------------------------------------------------------------------------------------------------------------------
TGJets_postEE                 = ak.from_parquet(f"{postEE_basepath}/Top/TGJets_postEE/nominal")
TTG_1Jets_PTG_10to100_postEE  = ak.from_parquet(f"{postEE_basepath}/Top/TTG-1Jets_PTG-10to100_postEE/nominal")
TTG_1Jets_PTG_100to200_postEE = ak.from_parquet(f"{postEE_basepath}/Top/TTG-1Jets_PTG-100to200_postEE/nominal")
TTG_1Jets_PTG_200_postEE      = ak.from_parquet(f"{postEE_basepath}/Top/TTG-1Jets_PTG-200_postEE/nominal")
TTGG_0Jets_postEE             = ak.from_parquet(f"{postEE_basepath}/Top/TTGG_0Jets_postEE/nominal")
TTto2L2Nu_2Jets_postEE        = ak.from_parquet(f"{postEE_basepath}/Top/TTto2L2Nu-2Jets_postEE/nominal")
TTto4Q_2Jets_postEE           = ak.from_parquet(f"{postEE_basepath}/Top/TTto4Q-2Jets_postEE/nominal")
TTtoLminusNu2Q_2Jets_postEE   = ak.from_parquet(f"{postEE_basepath}/Top/TTtoLminusNu2Q-2Jets_postEE/nominal")
TTtoLplusNu2Q_2Jets_postEE    = ak.from_parquet(f"{postEE_basepath}/Top/TTtoLplusNu2Q-2Jets_postEE/nominal")
print("[INFO] Ntuples of 2022 postEE Top quarks samples are loaded.")
########################
########## WG ##########
########################
WGtoLNuG_1Jets_PTG_10to100_preEE   = ak.from_parquet(f"{preEE_basepath}/WG/WGtoLNuG-1Jets_PTG-10to100_preEE/nominal")
WGtoLNuG_1Jets_PTG_100to200_preEE  = ak.from_parquet(f"{preEE_basepath}/WG/WGtoLNuG-1Jets_PTG-100to200_preEE/nominal")
WGtoLNuG_1Jets_PTG_200to400_preEE  = ak.from_parquet(f"{preEE_basepath}/WG/WGtoLNuG-1Jets_PTG-200to400_preEE/nominal")
WGtoLNuG_1Jets_PTG_400to600_preEE  = ak.from_parquet(f"{preEE_basepath}/WG/WGtoLNuG-1Jets_PTG-400to600_preEE/nominal")
WGtoLNuG_1Jets_PTG_600_preEE       = ak.from_parquet(f"{preEE_basepath}/WG/WGtoLNuG-1Jets_PTG-600_preEE/nominal")
print("[INFO] Ntuples of 2022 preEE WG samples are loaded.")
# ------------------------------------------------------------------------------------------------------------------------------
WGtoLNuG_1Jets_PTG_10to100_postEE  = ak.from_parquet(f"{postEE_basepath}/WG/WGtoLNuG-1Jets_PTG-10to100_postEE/nominal")
WGtoLNuG_1Jets_PTG_100to200_postEE = ak.from_parquet(f"{postEE_basepath}/WG/WGtoLNuG-1Jets_PTG-100to200_postEE/nominal")
WGtoLNuG_1Jets_PTG_200to400_postEE = ak.from_parquet(f"{postEE_basepath}/WG/WGtoLNuG-1Jets_PTG-200to400_postEE/nominal")
WGtoLNuG_1Jets_PTG_400to600_postEE = ak.from_parquet(f"{postEE_basepath}/WG/WGtoLNuG-1Jets_PTG-400to600_postEE/nominal")
WGtoLNuG_1Jets_PTG_600_postEE      = ak.from_parquet(f"{postEE_basepath}/WG/WGtoLNuG-1Jets_PTG-600_postEE/nominal")
print("[INFO] Ntuples of 2022 postEE WG samples are loaded.")
########################
########## ZG ##########
########################
ZGto2LG_1Jets_preEE                = ak.from_parquet(f"{preEE_basepath}/ZG/ZGto2LG-1Jets_preEE/nominal")
ZGto2NuG_1Jets_PTG_10to100_preEE   = ak.from_parquet(f"{preEE_basepath}/ZG/ZGto2NuG-1Jets_PTG-10to100_preEE/nominal")
ZGto2NuG_1Jets_PTG_100to200_preEE  = ak.from_parquet(f"{preEE_basepath}/ZG/ZGto2NuG-1Jets_PTG-100to200_preEE/nominal")
ZGto2NuG_1Jets_PTG_200to400_preEE  = ak.from_parquet(f"{preEE_basepath}/ZG/ZGto2NuG-1Jets_PTG-200to400_preEE/nominal")
ZGto2NuG_1Jets_PTG_400to600_preEE  = ak.from_parquet(f"{preEE_basepath}/ZG/ZGto2NuG-1Jets_PTG-400to600_preEE/nominal")
ZGto2NuG_1Jets_PTG_600_preEE       = ak.from_parquet(f"{preEE_basepath}/ZG/ZGto2NuG-1Jets_PTG-600_preEE/nominal")
print("[INFO] Ntuples of 2022 preEE ZG samples are loaded.")
# ------------------------------------------------------------------------------------------------------------------------------
ZGto2LG_1Jets_postEE               = ak.from_parquet(f"{postEE_basepath}/ZG/ZGto2LG-1Jets_postEE/nominal")
ZGto2NuG_1Jets_PTG_10to100_postEE  = ak.from_parquet(f"{postEE_basepath}/ZG/ZGto2NuG-1Jets_PTG-10to100_postEE/nominal")
ZGto2NuG_1Jets_PTG_100to200_postEE = ak.from_parquet(f"{postEE_basepath}/ZG/ZGto2NuG-1Jets_PTG-100to200_postEE/nominal")
ZGto2NuG_1Jets_PTG_200to400_postEE = ak.from_parquet(f"{postEE_basepath}/ZG/ZGto2NuG-1Jets_PTG-200to400_postEE/nominal")
ZGto2NuG_1Jets_PTG_400to600_postEE = ak.from_parquet(f"{postEE_basepath}/ZG/ZGto2NuG-1Jets_PTG-400to600_postEE/nominal")
ZGto2NuG_1Jets_PTG_600_postEE      = ak.from_parquet(f"{postEE_basepath}/ZG/ZGto2NuG-1Jets_PTG-600_postEE/nominal")
print("[INFO] Ntuples of 2022 postEE ZG samples are loaded.")
###############################
########## WH-signal ##########
###############################
WminusH_Hto2G_WtoLNu_M_125_preEE   = ak.from_parquet(f"{preEE_basepath}/WH_signal/WminusH_Hto2G_WtoLNu_M-125_powheg_preEE/nominal")
WplusH_Hto2G_WtoLNu_M_125_preEE    = ak.from_parquet(f"{preEE_basepath}/WH_signal/WplusH_Hto2G_WtoLNu_M-125_powheg_preEE/nominal")
# ------------------------------------------------------------------------------------------------------------------------------
WminusH_Hto2G_WtoLNu_M_125_postEE  = ak.from_parquet(f"{postEE_basepath}/WH_signal/WminusH_Hto2G_WtoLNu_M-125_powheg_postEE/nominal")
WplusH_Hto2G_WtoLNu_M_125_postEE   = ak.from_parquet(f"{postEE_basepath}/WH_signal/WplusH_Hto2G_WtoLNu_M-125_powheg_postEE/nominal")
################################################################################################################################
########## Keep the diphoton mass within 100-180 GeV for both Data and MC ##########
################################################################################################################################
##########################
########## Data ##########
##########################
DataC_preEE  = DataC_preEE[(DataC_preEE.mass > 100) & (DataC_preEE.mass < 180)]
DataD_preEE  = DataD_preEE[(DataD_preEE.mass > 100) & (DataD_preEE.mass < 180)]
# ------------------------------------------------------------------------------------------------------------------------------
DataE_postEE = DataE_postEE[(DataE_postEE.mass > 100) & (DataE_postEE.mass < 180)]
DataF_postEE = DataF_postEE[(DataF_postEE.mass > 100) & (DataF_postEE.mass < 180)]
DataG_postEE = DataG_postEE[(DataG_postEE.mass > 100) & (DataG_postEE.mass < 180)]
########################
########## VV ##########
########################
WW_preEE  = WW_preEE[(WW_preEE.mass > 100) & (WW_preEE.mass < 180)]
WZ_preEE  = WZ_preEE[(WZ_preEE.mass > 100) & (WZ_preEE.mass < 180)]
ZZ_preEE  = ZZ_preEE[(ZZ_preEE.mass > 100) & (ZZ_preEE.mass < 180)]
# ------------------------------------------------------------------------------------------------------------------------------
WW_postEE = WW_postEE[(WW_postEE.mass > 100) & (WW_postEE.mass < 180)]
WZ_postEE = WZ_postEE[(WZ_postEE.mass > 100) & (WZ_postEE.mass < 180)]
ZZ_postEE = ZZ_postEE[(ZZ_postEE.mass > 100) & (ZZ_postEE.mass < 180)]
###########################################
########## Diphoton-nonresonance ##########
###########################################
Diphoton_preEE  = Diphoton_preEE[(Diphoton_preEE.mass > 100) & (Diphoton_preEE.mass < 180)]
# ------------------------------------------------------------------------------------------------------------------------------
Diphoton_postEE = Diphoton_postEE[(Diphoton_postEE.mass > 100) & (Diphoton_postEE.mass < 180)]
############################
########## DYJets ##########
############################
DYto2L_2Jets_MLL_50_preEE                 = DYto2L_2Jets_MLL_50_preEE[(DYto2L_2Jets_MLL_50_preEE.mass > 100) & (DYto2L_2Jets_MLL_50_preEE.mass < 180)]
DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE    = DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE[(DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE.mass > 100) & (DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE.mass < 180)]
DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE   = DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE[(DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE.mass > 100) & (DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE.mass < 180)]
DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE  = DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE[(DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE.mass > 100) & (DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE.mass < 180)]
DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE  = DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE[(DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE.mass > 100) & (DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE.mass < 180)]
DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE  = DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE[(DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE.mass > 100) & (DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE.mass < 180)]
DYGto2LG_1Jets_MLL_50_PTG_600_preEE       = DYGto2LG_1Jets_MLL_50_PTG_600_preEE[(DYGto2LG_1Jets_MLL_50_PTG_600_preEE.mass > 100) & (DYGto2LG_1Jets_MLL_50_PTG_600_preEE.mass < 180)]
# ------------------------------------------------------------------------------------------------------------------------------
DYto2L_2Jets_MLL_50_postEE                = DYto2L_2Jets_MLL_50_postEE[(DYto2L_2Jets_MLL_50_postEE.mass > 100) & (DYto2L_2Jets_MLL_50_postEE.mass < 180)]
DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE   = DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE[(DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE.mass > 100) & (DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE.mass < 180)]
DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE  = DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE[(DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE.mass > 100) & (DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE.mass < 180)]
DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE = DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE[(DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE.mass > 100) & (DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE.mass < 180)]
DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE = DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE[(DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE.mass > 100) & (DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE.mass < 180)]
DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE = DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE[(DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE.mass > 100) & (DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE.mass < 180)]
DYGto2LG_1Jets_MLL_50_PTG_600_postEE      = DYGto2LG_1Jets_MLL_50_PTG_600_postEE[(DYGto2LG_1Jets_MLL_50_PTG_600_postEE.mass > 100) & (DYGto2LG_1Jets_MLL_50_PTG_600_postEE.mass < 180)]
###########################
########## GJets ##########
###########################
GJet_PT_20_MGG_40to80_preEE  = GJet_PT_20_MGG_40to80_preEE[(GJet_PT_20_MGG_40to80_preEE.mass > 100) & (GJet_PT_20_MGG_40to80_preEE.mass < 180)]
GJet_PT_20to40_MGG_80_preEE  = GJet_PT_20to40_MGG_80_preEE[(GJet_PT_20to40_MGG_80_preEE.mass > 100) & (GJet_PT_20to40_MGG_80_preEE.mass < 180)]
GJet_PT_40_MGG_80_preEE      = GJet_PT_40_MGG_80_preEE[(GJet_PT_40_MGG_80_preEE.mass > 100) & (GJet_PT_40_MGG_80_preEE.mass < 180)]
# ------------------------------------------------------------------------------------------------------------------------------
GJet_PT_20_MGG_40to80_postEE = GJet_PT_20_MGG_40to80_postEE[(GJet_PT_20_MGG_40to80_postEE.mass > 100) & (GJet_PT_20_MGG_40to80_postEE.mass < 180)]
GJet_PT_20to40_MGG_80_postEE = GJet_PT_20to40_MGG_80_postEE[(GJet_PT_20to40_MGG_80_postEE.mass > 100) & (GJet_PT_20to40_MGG_80_postEE.mass < 180)]
GJet_PT_40_MGG_80_postEE     = GJet_PT_40_MGG_80_postEE[(GJet_PT_40_MGG_80_postEE.mass > 100) & (GJet_PT_40_MGG_80_postEE.mass < 180)]
#########################
########## QCD ##########
#########################
QCD_PT_30to40_MGG_80toInf_preEE   = QCD_PT_30to40_MGG_80toInf_preEE[(QCD_PT_30to40_MGG_80toInf_preEE.mass > 100) & (QCD_PT_30to40_MGG_80toInf_preEE.mass < 180)]
QCD_PT_40toInf_MGG_80toInf_preEE  = QCD_PT_40toInf_MGG_80toInf_preEE[(QCD_PT_40toInf_MGG_80toInf_preEE.mass > 100) & (QCD_PT_40toInf_MGG_80toInf_preEE.mass < 180)]
# ------------------------------------------------------------------------------------------------------------------------------
QCD_PT_30to40_MGG_80toInf_postEE  = QCD_PT_30to40_MGG_80toInf_postEE[(QCD_PT_30to40_MGG_80toInf_postEE.mass > 100) & (QCD_PT_30to40_MGG_80toInf_postEE.mass < 180)]
QCD_PT_40toInf_MGG_80toInf_postEE = QCD_PT_40toInf_MGG_80toInf_postEE[(QCD_PT_40toInf_MGG_80toInf_postEE.mass > 100) & (QCD_PT_40toInf_MGG_80toInf_postEE.mass < 180)]
#########################
########## Top ##########
#########################
TGJets_preEE                  = TGJets_preEE[(TGJets_preEE.mass > 100) & (TGJets_preEE.mass < 180)]
TTG_1Jets_PTG_10to100_preEE   = TTG_1Jets_PTG_10to100_preEE[(TTG_1Jets_PTG_10to100_preEE.mass > 100) & (TTG_1Jets_PTG_10to100_preEE.mass < 180)]
TTG_1Jets_PTG_100to200_preEE  = TTG_1Jets_PTG_100to200_preEE[(TTG_1Jets_PTG_100to200_preEE.mass > 100) & (TTG_1Jets_PTG_100to200_preEE.mass < 180)]
TTG_1Jets_PTG_200_preEE       = TTG_1Jets_PTG_200_preEE[(TTG_1Jets_PTG_200_preEE.mass > 100) & (TTG_1Jets_PTG_200_preEE.mass < 180)]
TTGG_0Jets_preEE              = TTGG_0Jets_preEE[(TTGG_0Jets_preEE.mass > 100) & (TTGG_0Jets_preEE.mass < 180)]
TTto2L2Nu_2Jets_preEE         = TTto2L2Nu_2Jets_preEE[(TTto2L2Nu_2Jets_preEE.mass > 100) & (TTto2L2Nu_2Jets_preEE.mass < 180)]
TTto4Q_2Jets_preEE            = TTto4Q_2Jets_preEE[(TTto4Q_2Jets_preEE.mass > 100) & (TTto4Q_2Jets_preEE.mass < 180)]
TTtoLminusNu2Q_2Jets_preEE    = TTtoLminusNu2Q_2Jets_preEE[(TTtoLminusNu2Q_2Jets_preEE.mass > 100) & (TTtoLminusNu2Q_2Jets_preEE.mass < 180)]
TTtoLplusNu2Q_2Jets_preEE     = TTtoLplusNu2Q_2Jets_preEE[(TTtoLplusNu2Q_2Jets_preEE.mass > 100) & (TTtoLplusNu2Q_2Jets_preEE.mass < 180)]
# ------------------------------------------------------------------------------------------------------------------------------
TGJets_postEE                 = TGJets_postEE[(TGJets_postEE.mass > 100) & (TGJets_postEE.mass < 180)]
TTG_1Jets_PTG_10to100_postEE  = TTG_1Jets_PTG_10to100_postEE[(TTG_1Jets_PTG_10to100_postEE.mass > 100) & (TTG_1Jets_PTG_10to100_postEE.mass < 180)]
TTG_1Jets_PTG_100to200_postEE = TTG_1Jets_PTG_100to200_postEE[(TTG_1Jets_PTG_100to200_postEE.mass > 100) & (TTG_1Jets_PTG_100to200_postEE.mass < 180)]
TTG_1Jets_PTG_200_postEE      = TTG_1Jets_PTG_200_postEE[(TTG_1Jets_PTG_200_postEE.mass > 100) & (TTG_1Jets_PTG_200_postEE.mass < 180)]
TTGG_0Jets_postEE             = TTGG_0Jets_postEE[(TTGG_0Jets_postEE.mass > 100) & (TTGG_0Jets_postEE.mass < 180)]
TTto2L2Nu_2Jets_postEE        = TTto2L2Nu_2Jets_postEE[(TTto2L2Nu_2Jets_postEE.mass > 100) & (TTto2L2Nu_2Jets_postEE.mass < 180)]
TTto4Q_2Jets_postEE           = TTto4Q_2Jets_postEE[(TTto4Q_2Jets_postEE.mass > 100) & (TTto4Q_2Jets_postEE.mass < 180)]
TTtoLminusNu2Q_2Jets_postEE   = TTtoLminusNu2Q_2Jets_postEE[(TTtoLminusNu2Q_2Jets_postEE.mass > 100) & (TTtoLminusNu2Q_2Jets_postEE.mass < 180)]
TTtoLplusNu2Q_2Jets_postEE    = TTtoLplusNu2Q_2Jets_postEE[(TTtoLplusNu2Q_2Jets_postEE.mass > 100) & (TTtoLplusNu2Q_2Jets_postEE.mass < 180)]
########################
########## WG ##########
########################
WGtoLNuG_1Jets_PTG_10to100_preEE   = WGtoLNuG_1Jets_PTG_10to100_preEE[(WGtoLNuG_1Jets_PTG_10to100_preEE.mass > 100) & (WGtoLNuG_1Jets_PTG_10to100_preEE.mass < 180)]
WGtoLNuG_1Jets_PTG_100to200_preEE  = WGtoLNuG_1Jets_PTG_100to200_preEE[(WGtoLNuG_1Jets_PTG_100to200_preEE.mass > 100) & (WGtoLNuG_1Jets_PTG_100to200_preEE.mass < 180)]
WGtoLNuG_1Jets_PTG_200to400_preEE  = WGtoLNuG_1Jets_PTG_200to400_preEE[(WGtoLNuG_1Jets_PTG_200to400_preEE.mass > 100) & (WGtoLNuG_1Jets_PTG_200to400_preEE.mass < 180)]
WGtoLNuG_1Jets_PTG_400to600_preEE  = WGtoLNuG_1Jets_PTG_400to600_preEE[(WGtoLNuG_1Jets_PTG_400to600_preEE.mass > 100) & (WGtoLNuG_1Jets_PTG_400to600_preEE.mass < 180)]
WGtoLNuG_1Jets_PTG_600_preEE       = WGtoLNuG_1Jets_PTG_600_preEE[(WGtoLNuG_1Jets_PTG_600_preEE.mass > 100) & (WGtoLNuG_1Jets_PTG_600_preEE.mass < 180)]
# ------------------------------------------------------------------------------------------------------------------------------
WGtoLNuG_1Jets_PTG_10to100_postEE  = WGtoLNuG_1Jets_PTG_10to100_postEE[(WGtoLNuG_1Jets_PTG_10to100_postEE.mass > 100) & (WGtoLNuG_1Jets_PTG_10to100_postEE.mass < 180)]
WGtoLNuG_1Jets_PTG_100to200_postEE = WGtoLNuG_1Jets_PTG_100to200_postEE[(WGtoLNuG_1Jets_PTG_100to200_postEE.mass > 100) & (WGtoLNuG_1Jets_PTG_100to200_postEE.mass < 180)]
WGtoLNuG_1Jets_PTG_200to400_postEE = WGtoLNuG_1Jets_PTG_200to400_postEE[(WGtoLNuG_1Jets_PTG_200to400_postEE.mass > 100) & (WGtoLNuG_1Jets_PTG_200to400_postEE.mass < 180)]
WGtoLNuG_1Jets_PTG_400to600_postEE = WGtoLNuG_1Jets_PTG_400to600_postEE[(WGtoLNuG_1Jets_PTG_400to600_postEE.mass > 100) & (WGtoLNuG_1Jets_PTG_400to600_postEE.mass < 180)]
WGtoLNuG_1Jets_PTG_600_postEE      = WGtoLNuG_1Jets_PTG_600_postEE[(WGtoLNuG_1Jets_PTG_600_postEE.mass > 100) & (WGtoLNuG_1Jets_PTG_600_postEE.mass < 180)]
########################
########## ZG ##########
########################
ZGto2LG_1Jets_preEE                = ZGto2LG_1Jets_preEE[(ZGto2LG_1Jets_preEE.mass > 100) & (ZGto2LG_1Jets_preEE.mass < 180)]
ZGto2NuG_1Jets_PTG_10to100_preEE   = ZGto2NuG_1Jets_PTG_10to100_preEE[(ZGto2NuG_1Jets_PTG_10to100_preEE.mass > 100) & (ZGto2NuG_1Jets_PTG_10to100_preEE.mass < 180)]
ZGto2NuG_1Jets_PTG_100to200_preEE  = ZGto2NuG_1Jets_PTG_100to200_preEE[(ZGto2NuG_1Jets_PTG_100to200_preEE.mass > 100) & (ZGto2NuG_1Jets_PTG_100to200_preEE.mass < 180)]
ZGto2NuG_1Jets_PTG_200to400_preEE  = ZGto2NuG_1Jets_PTG_200to400_preEE[(ZGto2NuG_1Jets_PTG_200to400_preEE.mass > 100) & (ZGto2NuG_1Jets_PTG_200to400_preEE.mass < 180)]
ZGto2NuG_1Jets_PTG_400to600_preEE  = ZGto2NuG_1Jets_PTG_400to600_preEE[(ZGto2NuG_1Jets_PTG_400to600_preEE.mass > 100) & (ZGto2NuG_1Jets_PTG_400to600_preEE.mass < 180)]
ZGto2NuG_1Jets_PTG_600_preEE       = ZGto2NuG_1Jets_PTG_600_preEE[(ZGto2NuG_1Jets_PTG_600_preEE.mass > 100) & (ZGto2NuG_1Jets_PTG_600_preEE.mass < 180)]
# ------------------------------------------------------------------------------------------------------------------------------
ZGto2LG_1Jets_postEE               = ZGto2LG_1Jets_postEE[(ZGto2LG_1Jets_postEE.mass > 100) & (ZGto2LG_1Jets_postEE.mass < 180)]
ZGto2NuG_1Jets_PTG_10to100_postEE  = ZGto2NuG_1Jets_PTG_10to100_postEE[(ZGto2NuG_1Jets_PTG_10to100_postEE.mass > 100) & (ZGto2NuG_1Jets_PTG_10to100_postEE.mass < 180)]
ZGto2NuG_1Jets_PTG_100to200_postEE = ZGto2NuG_1Jets_PTG_100to200_postEE[(ZGto2NuG_1Jets_PTG_100to200_postEE.mass > 100) & (ZGto2NuG_1Jets_PTG_100to200_postEE.mass < 180)]
ZGto2NuG_1Jets_PTG_200to400_postEE = ZGto2NuG_1Jets_PTG_200to400_postEE[(ZGto2NuG_1Jets_PTG_200to400_postEE.mass > 100) & (ZGto2NuG_1Jets_PTG_200to400_postEE.mass < 180)]
ZGto2NuG_1Jets_PTG_400to600_postEE = ZGto2NuG_1Jets_PTG_400to600_postEE[(ZGto2NuG_1Jets_PTG_400to600_postEE.mass > 100) & (ZGto2NuG_1Jets_PTG_400to600_postEE.mass < 180)]
ZGto2NuG_1Jets_PTG_600_postEE      = ZGto2NuG_1Jets_PTG_600_postEE[(ZGto2NuG_1Jets_PTG_600_postEE.mass > 100) & (ZGto2NuG_1Jets_PTG_600_postEE.mass < 180)]
###############################
########## WH-signal ##########
###############################
WminusH_Hto2G_WtoLNu_M_125_preEE  = WminusH_Hto2G_WtoLNu_M_125_preEE[(WminusH_Hto2G_WtoLNu_M_125_preEE.mass > 100) | (WminusH_Hto2G_WtoLNu_M_125_preEE.mass < 180)]
WplusH_Hto2G_WtoLNu_M_125_preEE   = WplusH_Hto2G_WtoLNu_M_125_preEE[(WplusH_Hto2G_WtoLNu_M_125_preEE.mass > 100) | (WplusH_Hto2G_WtoLNu_M_125_preEE.mass < 180)]
# ------------------------------------------------------------------------------------------------------------------------------
WminusH_Hto2G_WtoLNu_M_125_postEE = WminusH_Hto2G_WtoLNu_M_125_postEE[(WminusH_Hto2G_WtoLNu_M_125_postEE.mass > 100) | (WminusH_Hto2G_WtoLNu_M_125_postEE.mass < 180)]
WplusH_Hto2G_WtoLNu_M_125_postEE  = WplusH_Hto2G_WtoLNu_M_125_postEE[(WplusH_Hto2G_WtoLNu_M_125_postEE.mass > 100) | (WplusH_Hto2G_WtoLNu_M_125_postEE.mass < 180)]
################################################################################################################################
########## Blind the signal region (115-135 GeV for diphoton mass) of both Data and MC ##########
################################################################################################################################
##########################
########## Data ##########
##########################
DataC_preEE  = DataC_preEE[(DataC_preEE.mass < 115) | (DataC_preEE.mass > 135)]
DataD_preEE  = DataD_preEE[(DataD_preEE.mass < 115) | (DataD_preEE.mass > 135)]
# ------------------------------------------------------------------------------------------------------------------------------
DataE_postEE = DataE_postEE[(DataE_postEE.mass < 115) | (DataE_postEE.mass > 135)]
DataF_postEE = DataF_postEE[(DataF_postEE.mass < 115) | (DataF_postEE.mass > 135)]
DataG_postEE = DataG_postEE[(DataG_postEE.mass < 115) | (DataG_postEE.mass > 135)]
########################
########## VV ##########
########################
WW_preEE  = WW_preEE[(WW_preEE.mass < 115) | (WW_preEE.mass > 135)]
WZ_preEE  = WZ_preEE[(WZ_preEE.mass < 115) | (WZ_preEE.mass > 135)]
ZZ_preEE  = ZZ_preEE[(ZZ_preEE.mass < 115) | (ZZ_preEE.mass > 135)]
# ------------------------------------------------------------------------------------------------------------------------------
WW_postEE = WW_postEE[(WW_postEE.mass < 115) | (WW_postEE.mass > 135)]
WZ_postEE = WZ_postEE[(WZ_postEE.mass < 115) | (WZ_postEE.mass > 135)]
ZZ_postEE = ZZ_postEE[(ZZ_postEE.mass < 115) | (ZZ_postEE.mass > 135)]
###########################################
########## Diphoton-nonresonance ##########
###########################################
Diphoton_preEE  = Diphoton_preEE[(Diphoton_preEE.mass < 115) | (Diphoton_preEE.mass > 135)]
# ------------------------------------------------------------------------------------------------------------------------------
Diphoton_postEE = Diphoton_postEE[(Diphoton_postEE.mass < 115) | (Diphoton_postEE.mass > 135)]
############################
########## DYJets ##########
############################
DYto2L_2Jets_MLL_50_preEE                 = DYto2L_2Jets_MLL_50_preEE[(DYto2L_2Jets_MLL_50_preEE.mass < 115) | (DYto2L_2Jets_MLL_50_preEE.mass > 135)]
DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE    = DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE[(DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE.mass < 115) | (DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE.mass > 135)]
DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE   = DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE[(DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE.mass < 115) | (DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE.mass > 135)]
DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE  = DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE[(DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE.mass < 115) | (DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE.mass > 135)]
DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE  = DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE[(DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE.mass < 115) | (DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE.mass > 135)]
DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE  = DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE[(DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE.mass < 115) | (DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE.mass > 135)]
DYGto2LG_1Jets_MLL_50_PTG_600_preEE       = DYGto2LG_1Jets_MLL_50_PTG_600_preEE[(DYGto2LG_1Jets_MLL_50_PTG_600_preEE.mass < 115) | (DYGto2LG_1Jets_MLL_50_PTG_600_preEE.mass > 135)]
# ------------------------------------------------------------------------------------------------------------------------------
DYto2L_2Jets_MLL_50_postEE                = DYto2L_2Jets_MLL_50_postEE[(DYto2L_2Jets_MLL_50_postEE.mass < 115) | (DYto2L_2Jets_MLL_50_postEE.mass > 135)]
DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE   = DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE[(DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE.mass < 115) | (DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE.mass > 135)]
DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE  = DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE[(DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE.mass < 115) | (DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE.mass > 135)]
DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE = DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE[(DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE.mass < 115) | (DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE.mass > 135)]
DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE = DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE[(DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE.mass < 115) | (DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE.mass > 135)]
DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE = DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE[(DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE.mass < 115) | (DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE.mass > 135)]
DYGto2LG_1Jets_MLL_50_PTG_600_postEE      = DYGto2LG_1Jets_MLL_50_PTG_600_postEE[(DYGto2LG_1Jets_MLL_50_PTG_600_postEE.mass < 115) | (DYGto2LG_1Jets_MLL_50_PTG_600_postEE.mass > 135)]
###########################
########## GJets ##########
###########################
GJet_PT_20_MGG_40to80_preEE  = GJet_PT_20_MGG_40to80_preEE[(GJet_PT_20_MGG_40to80_preEE.mass < 115) | (GJet_PT_20_MGG_40to80_preEE.mass > 135)]
GJet_PT_20to40_MGG_80_preEE  = GJet_PT_20to40_MGG_80_preEE[(GJet_PT_20to40_MGG_80_preEE.mass < 115) | (GJet_PT_20to40_MGG_80_preEE.mass > 135)]
GJet_PT_40_MGG_80_preEE      = GJet_PT_40_MGG_80_preEE[(GJet_PT_40_MGG_80_preEE.mass < 115) | (GJet_PT_40_MGG_80_preEE.mass > 135)]
# ------------------------------------------------------------------------------------------------------------------------------
GJet_PT_20_MGG_40to80_postEE = GJet_PT_20_MGG_40to80_postEE[(GJet_PT_20_MGG_40to80_postEE.mass < 115) | (GJet_PT_20_MGG_40to80_postEE.mass > 135)]
GJet_PT_20to40_MGG_80_postEE = GJet_PT_20to40_MGG_80_postEE[(GJet_PT_20to40_MGG_80_postEE.mass < 115) | (GJet_PT_20to40_MGG_80_postEE.mass > 135)]
GJet_PT_40_MGG_80_postEE     = GJet_PT_40_MGG_80_postEE[(GJet_PT_40_MGG_80_postEE.mass < 115) | (GJet_PT_40_MGG_80_postEE.mass > 135)]
#########################
########## QCD ##########
#########################
QCD_PT_30to40_MGG_80toInf_preEE   = QCD_PT_30to40_MGG_80toInf_preEE[(QCD_PT_30to40_MGG_80toInf_preEE.mass < 115) | (QCD_PT_30to40_MGG_80toInf_preEE.mass > 135)]
QCD_PT_40toInf_MGG_80toInf_preEE  = QCD_PT_40toInf_MGG_80toInf_preEE[(QCD_PT_40toInf_MGG_80toInf_preEE.mass < 115) | (QCD_PT_40toInf_MGG_80toInf_preEE.mass > 135)]
# ------------------------------------------------------------------------------------------------------------------------------
QCD_PT_30to40_MGG_80toInf_postEE  = QCD_PT_30to40_MGG_80toInf_postEE[(QCD_PT_30to40_MGG_80toInf_postEE.mass < 115) | (QCD_PT_30to40_MGG_80toInf_postEE.mass > 135)]
QCD_PT_40toInf_MGG_80toInf_postEE = QCD_PT_40toInf_MGG_80toInf_postEE[(QCD_PT_40toInf_MGG_80toInf_postEE.mass < 115) | (QCD_PT_40toInf_MGG_80toInf_postEE.mass > 135)]
#########################
########## Top ##########
#########################
TGJets_preEE                  = TGJets_preEE[(TGJets_preEE.mass < 115) | (TGJets_preEE.mass > 135)]
TTG_1Jets_PTG_10to100_preEE   = TTG_1Jets_PTG_10to100_preEE[(TTG_1Jets_PTG_10to100_preEE.mass < 115) | (TTG_1Jets_PTG_10to100_preEE.mass > 135)]
TTG_1Jets_PTG_100to200_preEE  = TTG_1Jets_PTG_100to200_preEE[(TTG_1Jets_PTG_100to200_preEE.mass < 115) | (TTG_1Jets_PTG_100to200_preEE.mass > 135)]
TTG_1Jets_PTG_200_preEE       = TTG_1Jets_PTG_200_preEE[(TTG_1Jets_PTG_200_preEE.mass < 115) | (TTG_1Jets_PTG_200_preEE.mass > 135)]
TTGG_0Jets_preEE              = TTGG_0Jets_preEE[(TTGG_0Jets_preEE.mass < 115) | (TTGG_0Jets_preEE.mass > 135)]
TTto2L2Nu_2Jets_preEE         = TTto2L2Nu_2Jets_preEE[(TTto2L2Nu_2Jets_preEE.mass < 115) | (TTto2L2Nu_2Jets_preEE.mass > 135)]
TTto4Q_2Jets_preEE            = TTto4Q_2Jets_preEE[(TTto4Q_2Jets_preEE.mass < 115) | (TTto4Q_2Jets_preEE.mass > 135)]
TTtoLminusNu2Q_2Jets_preEE    = TTtoLminusNu2Q_2Jets_preEE[(TTtoLminusNu2Q_2Jets_preEE.mass < 115) | (TTtoLminusNu2Q_2Jets_preEE.mass > 135)]
TTtoLplusNu2Q_2Jets_preEE     = TTtoLplusNu2Q_2Jets_preEE[(TTtoLplusNu2Q_2Jets_preEE.mass < 115) | (TTtoLplusNu2Q_2Jets_preEE.mass > 135)]
# ------------------------------------------------------------------------------------------------------------------------------
TGJets_postEE                 = TGJets_postEE[(TGJets_postEE.mass < 115) | (TGJets_postEE.mass > 135)]
TTG_1Jets_PTG_10to100_postEE  = TTG_1Jets_PTG_10to100_postEE[(TTG_1Jets_PTG_10to100_postEE.mass < 115) | (TTG_1Jets_PTG_10to100_postEE.mass > 135)]
TTG_1Jets_PTG_100to200_postEE = TTG_1Jets_PTG_100to200_postEE[(TTG_1Jets_PTG_100to200_postEE.mass < 115) | (TTG_1Jets_PTG_100to200_postEE.mass > 135)]
TTG_1Jets_PTG_200_postEE      = TTG_1Jets_PTG_200_postEE[(TTG_1Jets_PTG_200_postEE.mass < 115) | (TTG_1Jets_PTG_200_postEE.mass > 135)]
TTGG_0Jets_postEE             = TTGG_0Jets_postEE[(TTGG_0Jets_postEE.mass < 115) | (TTGG_0Jets_postEE.mass > 135)]
TTto2L2Nu_2Jets_postEE        = TTto2L2Nu_2Jets_postEE[(TTto2L2Nu_2Jets_postEE.mass < 115) | (TTto2L2Nu_2Jets_postEE.mass > 135)]
TTto4Q_2Jets_postEE           = TTto4Q_2Jets_postEE[(TTto4Q_2Jets_postEE.mass < 115) | (TTto4Q_2Jets_postEE.mass > 135)]
TTtoLminusNu2Q_2Jets_postEE   = TTtoLminusNu2Q_2Jets_postEE[(TTtoLminusNu2Q_2Jets_postEE.mass < 115) | (TTtoLminusNu2Q_2Jets_postEE.mass > 135)]
TTtoLplusNu2Q_2Jets_postEE    = TTtoLplusNu2Q_2Jets_postEE[(TTtoLplusNu2Q_2Jets_postEE.mass < 115) | (TTtoLplusNu2Q_2Jets_postEE.mass > 135)]
########################
########## WG ##########
########################
WGtoLNuG_1Jets_PTG_10to100_preEE   = WGtoLNuG_1Jets_PTG_10to100_preEE[(WGtoLNuG_1Jets_PTG_10to100_preEE.mass < 115) | (WGtoLNuG_1Jets_PTG_10to100_preEE.mass > 135)]
WGtoLNuG_1Jets_PTG_100to200_preEE  = WGtoLNuG_1Jets_PTG_100to200_preEE[(WGtoLNuG_1Jets_PTG_100to200_preEE.mass < 115) | (WGtoLNuG_1Jets_PTG_100to200_preEE.mass > 135)]
WGtoLNuG_1Jets_PTG_200to400_preEE  = WGtoLNuG_1Jets_PTG_200to400_preEE[(WGtoLNuG_1Jets_PTG_200to400_preEE.mass < 115) | (WGtoLNuG_1Jets_PTG_200to400_preEE.mass > 135)]
WGtoLNuG_1Jets_PTG_400to600_preEE  = WGtoLNuG_1Jets_PTG_400to600_preEE[(WGtoLNuG_1Jets_PTG_400to600_preEE.mass < 115) | (WGtoLNuG_1Jets_PTG_400to600_preEE.mass > 135)]
WGtoLNuG_1Jets_PTG_600_preEE       = WGtoLNuG_1Jets_PTG_600_preEE[(WGtoLNuG_1Jets_PTG_600_preEE.mass < 115) | (WGtoLNuG_1Jets_PTG_600_preEE.mass > 135)]
# ------------------------------------------------------------------------------------------------------------------------------
WGtoLNuG_1Jets_PTG_10to100_postEE  = WGtoLNuG_1Jets_PTG_10to100_postEE[(WGtoLNuG_1Jets_PTG_10to100_postEE.mass < 115) | (WGtoLNuG_1Jets_PTG_10to100_postEE.mass > 135)]
WGtoLNuG_1Jets_PTG_100to200_postEE = WGtoLNuG_1Jets_PTG_100to200_postEE[(WGtoLNuG_1Jets_PTG_100to200_postEE.mass < 115) | (WGtoLNuG_1Jets_PTG_100to200_postEE.mass > 135)]
WGtoLNuG_1Jets_PTG_200to400_postEE = WGtoLNuG_1Jets_PTG_200to400_postEE[(WGtoLNuG_1Jets_PTG_200to400_postEE.mass < 115) | (WGtoLNuG_1Jets_PTG_200to400_postEE.mass > 135)]
WGtoLNuG_1Jets_PTG_400to600_postEE = WGtoLNuG_1Jets_PTG_400to600_postEE[(WGtoLNuG_1Jets_PTG_400to600_postEE.mass < 115) | (WGtoLNuG_1Jets_PTG_400to600_postEE.mass > 135)]
WGtoLNuG_1Jets_PTG_600_postEE      = WGtoLNuG_1Jets_PTG_600_postEE[(WGtoLNuG_1Jets_PTG_600_postEE.mass < 115) | (WGtoLNuG_1Jets_PTG_600_postEE.mass > 135)]
########################
########## ZG ##########
########################
ZGto2LG_1Jets_preEE                = ZGto2LG_1Jets_preEE[(ZGto2LG_1Jets_preEE.mass < 115) | (ZGto2LG_1Jets_preEE.mass > 135)]
ZGto2NuG_1Jets_PTG_10to100_preEE   = ZGto2NuG_1Jets_PTG_10to100_preEE[(ZGto2NuG_1Jets_PTG_10to100_preEE.mass < 115) | (ZGto2NuG_1Jets_PTG_10to100_preEE.mass > 135)]
ZGto2NuG_1Jets_PTG_100to200_preEE  = ZGto2NuG_1Jets_PTG_100to200_preEE[(ZGto2NuG_1Jets_PTG_100to200_preEE.mass < 115) | (ZGto2NuG_1Jets_PTG_100to200_preEE.mass > 135)]
ZGto2NuG_1Jets_PTG_200to400_preEE  = ZGto2NuG_1Jets_PTG_200to400_preEE[(ZGto2NuG_1Jets_PTG_200to400_preEE.mass < 115) | (ZGto2NuG_1Jets_PTG_200to400_preEE.mass > 135)]
ZGto2NuG_1Jets_PTG_400to600_preEE  = ZGto2NuG_1Jets_PTG_400to600_preEE[(ZGto2NuG_1Jets_PTG_400to600_preEE.mass < 115) | (ZGto2NuG_1Jets_PTG_400to600_preEE.mass > 135)]
ZGto2NuG_1Jets_PTG_600_preEE       = ZGto2NuG_1Jets_PTG_600_preEE[(ZGto2NuG_1Jets_PTG_600_preEE.mass < 115) | (ZGto2NuG_1Jets_PTG_600_preEE.mass > 135)]
# ------------------------------------------------------------------------------------------------------------------------------
ZGto2LG_1Jets_postEE               = ZGto2LG_1Jets_postEE[(ZGto2LG_1Jets_postEE.mass < 115) | (ZGto2LG_1Jets_postEE.mass > 135)]
ZGto2NuG_1Jets_PTG_10to100_postEE  = ZGto2NuG_1Jets_PTG_10to100_postEE[(ZGto2NuG_1Jets_PTG_10to100_postEE.mass < 115) | (ZGto2NuG_1Jets_PTG_10to100_postEE.mass > 135)]
ZGto2NuG_1Jets_PTG_100to200_postEE = ZGto2NuG_1Jets_PTG_100to200_postEE[(ZGto2NuG_1Jets_PTG_100to200_postEE.mass < 115) | (ZGto2NuG_1Jets_PTG_100to200_postEE.mass > 135)]
ZGto2NuG_1Jets_PTG_200to400_postEE = ZGto2NuG_1Jets_PTG_200to400_postEE[(ZGto2NuG_1Jets_PTG_200to400_postEE.mass < 115) | (ZGto2NuG_1Jets_PTG_200to400_postEE.mass > 135)]
ZGto2NuG_1Jets_PTG_400to600_postEE = ZGto2NuG_1Jets_PTG_400to600_postEE[(ZGto2NuG_1Jets_PTG_400to600_postEE.mass < 115) | (ZGto2NuG_1Jets_PTG_400to600_postEE.mass > 135)]
ZGto2NuG_1Jets_PTG_600_postEE      = ZGto2NuG_1Jets_PTG_600_postEE[(ZGto2NuG_1Jets_PTG_600_postEE.mass < 115) | (ZGto2NuG_1Jets_PTG_600_postEE.mass > 135)]
###############################
########## WH-signal ##########
###############################
# WminusH_Hto2G_WtoLNu_M_125_preEE  = WminusH_Hto2G_WtoLNu_M_125_preEE[(WminusH_Hto2G_WtoLNu_M_125_preEE.mass < 115) | (WminusH_Hto2G_WtoLNu_M_125_preEE.mass > 135)]
# WplusH_Hto2G_WtoLNu_M_125_preEE   = WplusH_Hto2G_WtoLNu_M_125_preEE[(WplusH_Hto2G_WtoLNu_M_125_preEE.mass < 115) | (WplusH_Hto2G_WtoLNu_M_125_preEE.mass > 135)]
# # ------------------------------------------------------------------------------------------------------------------------------
# WminusH_Hto2G_WtoLNu_M_125_postEE = WminusH_Hto2G_WtoLNu_M_125_postEE[(WminusH_Hto2G_WtoLNu_M_125_postEE.mass < 115) | (WminusH_Hto2G_WtoLNu_M_125_postEE.mass > 135)]
# WplusH_Hto2G_WtoLNu_M_125_postEE  = WplusH_Hto2G_WtoLNu_M_125_postEE[(WplusH_Hto2G_WtoLNu_M_125_postEE.mass < 115) | (WplusH_Hto2G_WtoLNu_M_125_postEE.mass > 135)]
################################################################################################################################
########## Require the mvaID score of both lead- and sublead-photons > -0.4 for both Data and MC ##########
################################################################################################################################
##########################
########## Data ##########
##########################
DataC_preEE  = DataC_preEE[(DataC_preEE.lead_mvaID > -0.4) & (DataC_preEE.sublead_mvaID > -0.4)]
DataD_preEE  = DataD_preEE[(DataD_preEE.lead_mvaID > -0.4) & (DataD_preEE.sublead_mvaID > -0.4)]
# ------------------------------------------------------------------------------------------------------------------------------
DataE_postEE = DataE_postEE[(DataE_postEE.lead_mvaID > -0.4) & (DataE_postEE.sublead_mvaID > -0.4)]
DataF_postEE = DataF_postEE[(DataF_postEE.lead_mvaID > -0.4) & (DataF_postEE.sublead_mvaID > -0.4)]
DataG_postEE = DataG_postEE[(DataG_postEE.lead_mvaID > -0.4) & (DataG_postEE.sublead_mvaID > -0.4)]
########################
########## VV ##########
########################
WW_preEE  = WW_preEE[(WW_preEE.lead_mvaID > -0.4) & (WW_preEE.sublead_mvaID > -0.4)]
WZ_preEE  = WZ_preEE[(WZ_preEE.lead_mvaID > -0.4) & (WZ_preEE.sublead_mvaID > -0.4)]
ZZ_preEE  = ZZ_preEE[(ZZ_preEE.lead_mvaID > -0.4) & (ZZ_preEE.sublead_mvaID > -0.4)]
# ------------------------------------------------------------------------------------------------------------------------------
WW_postEE = WW_postEE[(WW_postEE.lead_mvaID > -0.4) & (WW_postEE.sublead_mvaID > -0.4)]
WZ_postEE = WZ_postEE[(WZ_postEE.lead_mvaID > -0.4) & (WZ_postEE.sublead_mvaID > -0.4)]
ZZ_postEE = ZZ_postEE[(ZZ_postEE.lead_mvaID > -0.4) & (ZZ_postEE.sublead_mvaID > -0.4)]
###########################################
########## Diphoton-nonresonance ##########
###########################################
Diphoton_preEE  = Diphoton_preEE[(Diphoton_preEE.lead_mvaID > -0.4) & (Diphoton_preEE.sublead_mvaID > -0.4)]
# ------------------------------------------------------------------------------------------------------------------------------
Diphoton_postEE = Diphoton_postEE[(Diphoton_postEE.lead_mvaID > -0.4) & (Diphoton_postEE.sublead_mvaID > -0.4)]
############################
########## DYJets ##########
############################
DYto2L_2Jets_MLL_50_preEE                 = DYto2L_2Jets_MLL_50_preEE[(DYto2L_2Jets_MLL_50_preEE.lead_mvaID > -0.4) & (DYto2L_2Jets_MLL_50_preEE.sublead_mvaID > -0.4)]
DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE    = DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE[(DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE.lead_mvaID > -0.4) & (DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE.sublead_mvaID > -0.4)]
DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE   = DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE[(DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE.lead_mvaID > -0.4) & (DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE.sublead_mvaID > -0.4)]
DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE  = DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE[(DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE.lead_mvaID > -0.4) & (DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE.sublead_mvaID > -0.4)]
DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE  = DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE[(DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE.lead_mvaID > -0.4) & (DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE.sublead_mvaID > -0.4)]
DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE  = DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE[(DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE.lead_mvaID > -0.4) & (DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE.sublead_mvaID > -0.4)]
DYGto2LG_1Jets_MLL_50_PTG_600_preEE       = DYGto2LG_1Jets_MLL_50_PTG_600_preEE[(DYGto2LG_1Jets_MLL_50_PTG_600_preEE.lead_mvaID > -0.4) & (DYGto2LG_1Jets_MLL_50_PTG_600_preEE.sublead_mvaID > -0.4)]
# ------------------------------------------------------------------------------------------------------------------------------
DYto2L_2Jets_MLL_50_postEE                = DYto2L_2Jets_MLL_50_postEE[(DYto2L_2Jets_MLL_50_postEE.lead_mvaID > -0.4) & (DYto2L_2Jets_MLL_50_postEE.sublead_mvaID > -0.4)]
DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE   = DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE[(DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE.lead_mvaID > -0.4) & (DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE.sublead_mvaID > -0.4)]
DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE  = DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE[(DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE.lead_mvaID > -0.4) & (DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE.sublead_mvaID > -0.4)]
DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE = DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE[(DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE.lead_mvaID > -0.4) & (DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE.sublead_mvaID > -0.4)]
DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE = DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE[(DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE.lead_mvaID > -0.4) & (DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE.sublead_mvaID > -0.4)]
DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE = DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE[(DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE.lead_mvaID > -0.4) & (DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE.sublead_mvaID > -0.4)]
DYGto2LG_1Jets_MLL_50_PTG_600_postEE      = DYGto2LG_1Jets_MLL_50_PTG_600_postEE[(DYGto2LG_1Jets_MLL_50_PTG_600_postEE.lead_mvaID > -0.4) & (DYGto2LG_1Jets_MLL_50_PTG_600_postEE.sublead_mvaID > -0.4)]
###########################
########## GJets ##########
###########################
GJet_PT_20_MGG_40to80_preEE  = GJet_PT_20_MGG_40to80_preEE[(GJet_PT_20_MGG_40to80_preEE.lead_mvaID > -0.4) & (GJet_PT_20_MGG_40to80_preEE.sublead_mvaID > -0.4)]
GJet_PT_20to40_MGG_80_preEE  = GJet_PT_20to40_MGG_80_preEE[(GJet_PT_20to40_MGG_80_preEE.lead_mvaID > -0.4) & (GJet_PT_20to40_MGG_80_preEE.sublead_mvaID > -0.4)]
GJet_PT_40_MGG_80_preEE      = GJet_PT_40_MGG_80_preEE[(GJet_PT_40_MGG_80_preEE.lead_mvaID > -0.4) & (GJet_PT_40_MGG_80_preEE.sublead_mvaID > -0.4)]
# ------------------------------------------------------------------------------------------------------------------------------
GJet_PT_20_MGG_40to80_postEE = GJet_PT_20_MGG_40to80_postEE[(GJet_PT_20_MGG_40to80_postEE.lead_mvaID > -0.4) & (GJet_PT_20_MGG_40to80_postEE.sublead_mvaID > -0.4)]
GJet_PT_20to40_MGG_80_postEE = GJet_PT_20to40_MGG_80_postEE[(GJet_PT_20to40_MGG_80_postEE.lead_mvaID > -0.4) & (GJet_PT_20to40_MGG_80_postEE.sublead_mvaID > -0.4)]
GJet_PT_40_MGG_80_postEE     = GJet_PT_40_MGG_80_postEE[(GJet_PT_40_MGG_80_postEE.lead_mvaID > -0.4) & (GJet_PT_40_MGG_80_postEE.sublead_mvaID > -0.4)]
#########################
########## QCD ##########
#########################
QCD_PT_30to40_MGG_80toInf_preEE   = QCD_PT_30to40_MGG_80toInf_preEE[(QCD_PT_30to40_MGG_80toInf_preEE.lead_mvaID > -0.4) & (QCD_PT_30to40_MGG_80toInf_preEE.sublead_mvaID > -0.4)]
QCD_PT_40toInf_MGG_80toInf_preEE  = QCD_PT_40toInf_MGG_80toInf_preEE[(QCD_PT_40toInf_MGG_80toInf_preEE.lead_mvaID > -0.4) & (QCD_PT_40toInf_MGG_80toInf_preEE.sublead_mvaID > -0.4)]
# ------------------------------------------------------------------------------------------------------------------------------
QCD_PT_30to40_MGG_80toInf_postEE  = QCD_PT_30to40_MGG_80toInf_postEE[(QCD_PT_30to40_MGG_80toInf_postEE.lead_mvaID > -0.4) & (QCD_PT_30to40_MGG_80toInf_postEE.sublead_mvaID > -0.4)]
QCD_PT_40toInf_MGG_80toInf_postEE = QCD_PT_40toInf_MGG_80toInf_postEE[(QCD_PT_40toInf_MGG_80toInf_postEE.lead_mvaID > -0.4) & (QCD_PT_40toInf_MGG_80toInf_postEE.sublead_mvaID > -0.4)]
#########################
########## Top ##########
#########################
TGJets_preEE                  = TGJets_preEE[(TGJets_preEE.lead_mvaID > -0.4) & (TGJets_preEE.sublead_mvaID > -0.4)]
TTG_1Jets_PTG_10to100_preEE   = TTG_1Jets_PTG_10to100_preEE[(TTG_1Jets_PTG_10to100_preEE.lead_mvaID > -0.4) & (TTG_1Jets_PTG_10to100_preEE.sublead_mvaID > -0.4)]
TTG_1Jets_PTG_100to200_preEE  = TTG_1Jets_PTG_100to200_preEE[(TTG_1Jets_PTG_100to200_preEE.lead_mvaID > -0.4) & (TTG_1Jets_PTG_100to200_preEE.sublead_mvaID > -0.4)]
TTG_1Jets_PTG_200_preEE       = TTG_1Jets_PTG_200_preEE[(TTG_1Jets_PTG_200_preEE.lead_mvaID > -0.4) & (TTG_1Jets_PTG_200_preEE.sublead_mvaID > -0.4)]
TTGG_0Jets_preEE              = TTGG_0Jets_preEE[(TTGG_0Jets_preEE.lead_mvaID > -0.4) & (TTGG_0Jets_preEE.sublead_mvaID > -0.4)]
TTto2L2Nu_2Jets_preEE         = TTto2L2Nu_2Jets_preEE[(TTto2L2Nu_2Jets_preEE.lead_mvaID > -0.4) & (TTto2L2Nu_2Jets_preEE.sublead_mvaID > -0.4)]
TTto4Q_2Jets_preEE            = TTto4Q_2Jets_preEE[(TTto4Q_2Jets_preEE.lead_mvaID > -0.4) & (TTto4Q_2Jets_preEE.sublead_mvaID > -0.4)]
TTtoLminusNu2Q_2Jets_preEE    = TTtoLminusNu2Q_2Jets_preEE[(TTtoLminusNu2Q_2Jets_preEE.lead_mvaID > -0.4) & (TTtoLminusNu2Q_2Jets_preEE.sublead_mvaID > -0.4)]
TTtoLplusNu2Q_2Jets_preEE     = TTtoLplusNu2Q_2Jets_preEE[(TTtoLplusNu2Q_2Jets_preEE.lead_mvaID > -0.4) & (TTtoLplusNu2Q_2Jets_preEE.sublead_mvaID > -0.4)]
# ------------------------------------------------------------------------------------------------------------------------------
TGJets_postEE                 = TGJets_postEE[(TGJets_postEE.lead_mvaID > -0.4) & (TGJets_postEE.sublead_mvaID > -0.4)]
TTG_1Jets_PTG_10to100_postEE  = TTG_1Jets_PTG_10to100_postEE[(TTG_1Jets_PTG_10to100_postEE.lead_mvaID > -0.4) & (TTG_1Jets_PTG_10to100_postEE.sublead_mvaID > -0.4)]
TTG_1Jets_PTG_100to200_postEE = TTG_1Jets_PTG_100to200_postEE[(TTG_1Jets_PTG_100to200_postEE.lead_mvaID > -0.4) & (TTG_1Jets_PTG_100to200_postEE.sublead_mvaID > -0.4)]
TTG_1Jets_PTG_200_postEE      = TTG_1Jets_PTG_200_postEE[(TTG_1Jets_PTG_200_postEE.lead_mvaID > -0.4) & (TTG_1Jets_PTG_200_postEE.sublead_mvaID > -0.4)]
TTGG_0Jets_postEE             = TTGG_0Jets_postEE[(TTGG_0Jets_postEE.lead_mvaID > -0.4) & (TTGG_0Jets_postEE.sublead_mvaID > -0.4)]
TTto2L2Nu_2Jets_postEE        = TTto2L2Nu_2Jets_postEE[(TTto2L2Nu_2Jets_postEE.lead_mvaID > -0.4) & (TTto2L2Nu_2Jets_postEE.sublead_mvaID > -0.4)]
TTto4Q_2Jets_postEE           = TTto4Q_2Jets_postEE[(TTto4Q_2Jets_postEE.lead_mvaID > -0.4) & (TTto4Q_2Jets_postEE.sublead_mvaID > -0.4)]
TTtoLminusNu2Q_2Jets_postEE   = TTtoLminusNu2Q_2Jets_postEE[(TTtoLminusNu2Q_2Jets_postEE.lead_mvaID > -0.4) & (TTtoLminusNu2Q_2Jets_postEE.sublead_mvaID > -0.4)]
TTtoLplusNu2Q_2Jets_postEE    = TTtoLplusNu2Q_2Jets_postEE[(TTtoLplusNu2Q_2Jets_postEE.lead_mvaID > -0.4) & (TTtoLplusNu2Q_2Jets_postEE.sublead_mvaID > -0.4)]
########################
########## WG ##########
########################
WGtoLNuG_1Jets_PTG_10to100_preEE   = WGtoLNuG_1Jets_PTG_10to100_preEE[(WGtoLNuG_1Jets_PTG_10to100_preEE.lead_mvaID > -0.4) & (WGtoLNuG_1Jets_PTG_10to100_preEE.sublead_mvaID > -0.4)]
WGtoLNuG_1Jets_PTG_100to200_preEE  = WGtoLNuG_1Jets_PTG_100to200_preEE[(WGtoLNuG_1Jets_PTG_100to200_preEE.lead_mvaID > -0.4) & (WGtoLNuG_1Jets_PTG_100to200_preEE.sublead_mvaID > -0.4)]
WGtoLNuG_1Jets_PTG_200to400_preEE  = WGtoLNuG_1Jets_PTG_200to400_preEE[(WGtoLNuG_1Jets_PTG_200to400_preEE.lead_mvaID > -0.4) & (WGtoLNuG_1Jets_PTG_200to400_preEE.sublead_mvaID > -0.4)]
WGtoLNuG_1Jets_PTG_400to600_preEE  = WGtoLNuG_1Jets_PTG_400to600_preEE[(WGtoLNuG_1Jets_PTG_400to600_preEE.lead_mvaID > -0.4) & (WGtoLNuG_1Jets_PTG_400to600_preEE.sublead_mvaID > -0.4)]
WGtoLNuG_1Jets_PTG_600_preEE       = WGtoLNuG_1Jets_PTG_600_preEE[(WGtoLNuG_1Jets_PTG_600_preEE.lead_mvaID > -0.4) & (WGtoLNuG_1Jets_PTG_600_preEE.sublead_mvaID > -0.4)]
# ------------------------------------------------------------------------------------------------------------------------------
WGtoLNuG_1Jets_PTG_10to100_postEE  = WGtoLNuG_1Jets_PTG_10to100_postEE[(WGtoLNuG_1Jets_PTG_10to100_postEE.lead_mvaID > -0.4) & (WGtoLNuG_1Jets_PTG_10to100_postEE.sublead_mvaID > -0.4)]
WGtoLNuG_1Jets_PTG_100to200_postEE = WGtoLNuG_1Jets_PTG_100to200_postEE[(WGtoLNuG_1Jets_PTG_100to200_postEE.lead_mvaID > -0.4) & (WGtoLNuG_1Jets_PTG_100to200_postEE.sublead_mvaID > -0.4)]
WGtoLNuG_1Jets_PTG_200to400_postEE = WGtoLNuG_1Jets_PTG_200to400_postEE[(WGtoLNuG_1Jets_PTG_200to400_postEE.lead_mvaID > -0.4) & (WGtoLNuG_1Jets_PTG_200to400_postEE.sublead_mvaID > -0.4)]
WGtoLNuG_1Jets_PTG_400to600_postEE = WGtoLNuG_1Jets_PTG_400to600_postEE[(WGtoLNuG_1Jets_PTG_400to600_postEE.lead_mvaID > -0.4) & (WGtoLNuG_1Jets_PTG_400to600_postEE.sublead_mvaID > -0.4)]
WGtoLNuG_1Jets_PTG_600_postEE      = WGtoLNuG_1Jets_PTG_600_postEE[(WGtoLNuG_1Jets_PTG_600_postEE.lead_mvaID > -0.4) & (WGtoLNuG_1Jets_PTG_600_postEE.sublead_mvaID > -0.4)]
########################
########## ZG ##########
########################
ZGto2LG_1Jets_preEE                = ZGto2LG_1Jets_preEE[(ZGto2LG_1Jets_preEE.lead_mvaID > -0.4) & (ZGto2LG_1Jets_preEE.sublead_mvaID > -0.4)]
ZGto2NuG_1Jets_PTG_10to100_preEE   = ZGto2NuG_1Jets_PTG_10to100_preEE[(ZGto2NuG_1Jets_PTG_10to100_preEE.lead_mvaID > -0.4) & (ZGto2NuG_1Jets_PTG_10to100_preEE.sublead_mvaID > -0.4)]
ZGto2NuG_1Jets_PTG_100to200_preEE  = ZGto2NuG_1Jets_PTG_100to200_preEE[(ZGto2NuG_1Jets_PTG_100to200_preEE.lead_mvaID > -0.4) & (ZGto2NuG_1Jets_PTG_100to200_preEE.sublead_mvaID > -0.4)]
ZGto2NuG_1Jets_PTG_200to400_preEE  = ZGto2NuG_1Jets_PTG_200to400_preEE[(ZGto2NuG_1Jets_PTG_200to400_preEE.lead_mvaID > -0.4) & (ZGto2NuG_1Jets_PTG_200to400_preEE.sublead_mvaID > -0.4)]
ZGto2NuG_1Jets_PTG_400to600_preEE  = ZGto2NuG_1Jets_PTG_400to600_preEE[(ZGto2NuG_1Jets_PTG_400to600_preEE.lead_mvaID > -0.4) & (ZGto2NuG_1Jets_PTG_400to600_preEE.sublead_mvaID > -0.4)]
ZGto2NuG_1Jets_PTG_600_preEE       = ZGto2NuG_1Jets_PTG_600_preEE[(ZGto2NuG_1Jets_PTG_600_preEE.lead_mvaID > -0.4) & (ZGto2NuG_1Jets_PTG_600_preEE.sublead_mvaID > -0.4)]
# ------------------------------------------------------------------------------------------------------------------------------
ZGto2LG_1Jets_postEE               = ZGto2LG_1Jets_postEE[(ZGto2LG_1Jets_postEE.lead_mvaID > -0.4) & (ZGto2LG_1Jets_postEE.sublead_mvaID > -0.4)]
ZGto2NuG_1Jets_PTG_10to100_postEE  = ZGto2NuG_1Jets_PTG_10to100_postEE[(ZGto2NuG_1Jets_PTG_10to100_postEE.lead_mvaID > -0.4) & (ZGto2NuG_1Jets_PTG_10to100_postEE.sublead_mvaID > -0.4)]
ZGto2NuG_1Jets_PTG_100to200_postEE = ZGto2NuG_1Jets_PTG_100to200_postEE[(ZGto2NuG_1Jets_PTG_100to200_postEE.lead_mvaID > -0.4) & (ZGto2NuG_1Jets_PTG_100to200_postEE.sublead_mvaID > -0.4)]
ZGto2NuG_1Jets_PTG_200to400_postEE = ZGto2NuG_1Jets_PTG_200to400_postEE[(ZGto2NuG_1Jets_PTG_200to400_postEE.lead_mvaID > -0.4) & (ZGto2NuG_1Jets_PTG_200to400_postEE.sublead_mvaID > -0.4)]
ZGto2NuG_1Jets_PTG_400to600_postEE = ZGto2NuG_1Jets_PTG_400to600_postEE[(ZGto2NuG_1Jets_PTG_400to600_postEE.lead_mvaID > -0.4) & (ZGto2NuG_1Jets_PTG_400to600_postEE.sublead_mvaID > -0.4)]
ZGto2NuG_1Jets_PTG_600_postEE      = ZGto2NuG_1Jets_PTG_600_postEE[(ZGto2NuG_1Jets_PTG_600_postEE.lead_mvaID > -0.4) & (ZGto2NuG_1Jets_PTG_600_postEE.sublead_mvaID > -0.4)]
###############################
########## WH-signal ##########
###############################
WminusH_Hto2G_WtoLNu_M_125_preEE  = WminusH_Hto2G_WtoLNu_M_125_preEE[(WminusH_Hto2G_WtoLNu_M_125_preEE.lead_mvaID > -0.4) | (WminusH_Hto2G_WtoLNu_M_125_preEE.sublead_mvaID > -0.4)]
WplusH_Hto2G_WtoLNu_M_125_preEE   = WplusH_Hto2G_WtoLNu_M_125_preEE[(WplusH_Hto2G_WtoLNu_M_125_preEE.lead_mvaID > -0.4) | (WplusH_Hto2G_WtoLNu_M_125_preEE.sublead_mvaID > -0.4)]
# ------------------------------------------------------------------------------------------------------------------------------
WminusH_Hto2G_WtoLNu_M_125_postEE = WminusH_Hto2G_WtoLNu_M_125_postEE[(WminusH_Hto2G_WtoLNu_M_125_postEE.lead_mvaID > -0.4) | (WminusH_Hto2G_WtoLNu_M_125_postEE.sublead_mvaID > -0.4)]
WplusH_Hto2G_WtoLNu_M_125_postEE  = WplusH_Hto2G_WtoLNu_M_125_postEE[(WplusH_Hto2G_WtoLNu_M_125_postEE.lead_mvaID > -0.4) | (WplusH_Hto2G_WtoLNu_M_125_postEE.sublead_mvaID > -0.4)]
################################################################################################################################
########## Pick events with only 1 electron or 1 muon ##########
################################################################################################################################
##########################
########## Data ##########
##########################
DataC_preEE  = DataC_preEE[(DataC_preEE.n_electrons + DataC_preEE.n_muons) == 1]
DataD_preEE  = DataD_preEE[(DataD_preEE.n_electrons + DataD_preEE.n_muons) == 1]
# ------------------------------------------------------------------------------------------------------------------------------
DataE_postEE = DataE_postEE[(DataE_postEE.n_electrons + DataE_postEE.n_muons) == 1]
DataF_postEE = DataF_postEE[(DataF_postEE.n_electrons + DataF_postEE.n_muons) == 1]
DataG_postEE = DataG_postEE[(DataG_postEE.n_electrons + DataG_postEE.n_muons) == 1]
########################
########## VV ##########
########################
WW_preEE  = WW_preEE[(WW_preEE.n_electrons + WW_preEE.n_muons) == 1]
WZ_preEE  = WZ_preEE[(WZ_preEE.n_electrons + WZ_preEE.n_muons) == 1]
ZZ_preEE  = ZZ_preEE[(ZZ_preEE.n_electrons + ZZ_preEE.n_muons) == 1]
# ------------------------------------------------------------------------------------------------------------------------------
WW_postEE = WW_postEE[(WW_postEE.n_electrons + WW_postEE.n_muons) == 1]
WZ_postEE = WZ_postEE[(WZ_postEE.n_electrons + WZ_postEE.n_muons) == 1]
ZZ_postEE = ZZ_postEE[(ZZ_postEE.n_electrons + ZZ_postEE.n_muons) == 1]
###########################################
########## Diphoton-nonresonance ##########
###########################################
Diphoton_preEE  = Diphoton_preEE[(Diphoton_preEE.n_electrons + Diphoton_preEE.n_muons) == 1]
# ------------------------------------------------------------------------------------------------------------------------------
Diphoton_postEE = Diphoton_postEE[(Diphoton_postEE.n_electrons + Diphoton_postEE.n_muons) == 1]
############################
########## DYJets ##########
############################
DYto2L_2Jets_MLL_50_preEE                 = DYto2L_2Jets_MLL_50_preEE[(DYto2L_2Jets_MLL_50_preEE.n_electrons + DYto2L_2Jets_MLL_50_preEE.n_muons) == 1]
DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE    = DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE[(DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE.n_electrons + DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE.n_muons) == 1]
DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE   = DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE[(DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE.n_electrons + DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE.n_muons) == 1]
DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE  = DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE[(DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE.n_electrons + DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE.n_muons) == 1]
DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE  = DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE[(DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE.n_electrons + DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE.n_muons) == 1]
DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE  = DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE[(DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE.n_electrons + DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE.n_muons) == 1]
DYGto2LG_1Jets_MLL_50_PTG_600_preEE       = DYGto2LG_1Jets_MLL_50_PTG_600_preEE[(DYGto2LG_1Jets_MLL_50_PTG_600_preEE.n_electrons + DYGto2LG_1Jets_MLL_50_PTG_600_preEE.n_muons) == 1]
# ------------------------------------------------------------------------------------------------------------------------------
DYto2L_2Jets_MLL_50_postEE                = DYto2L_2Jets_MLL_50_postEE[(DYto2L_2Jets_MLL_50_postEE.n_electrons + DYto2L_2Jets_MLL_50_postEE.n_muons) == 1]
DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE   = DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE[(DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE.n_electrons + DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE.n_muons) == 1]
DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE  = DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE[(DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE.n_electrons + DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE.n_muons) == 1]
DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE = DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE[(DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE.n_electrons + DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE.n_muons) == 1]
DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE = DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE[(DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE.n_electrons + DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE.n_muons) == 1]
DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE = DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE[(DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE.n_electrons + DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE.n_muons) == 1]
DYGto2LG_1Jets_MLL_50_PTG_600_postEE      = DYGto2LG_1Jets_MLL_50_PTG_600_postEE[(DYGto2LG_1Jets_MLL_50_PTG_600_postEE.n_electrons + DYGto2LG_1Jets_MLL_50_PTG_600_postEE.n_muons) == 1]
###########################
########## GJets ##########
###########################
GJet_PT_20_MGG_40to80_preEE  = GJet_PT_20_MGG_40to80_preEE[(GJet_PT_20_MGG_40to80_preEE.n_electrons + GJet_PT_20_MGG_40to80_preEE.n_muons) == 1]
GJet_PT_20to40_MGG_80_preEE  = GJet_PT_20to40_MGG_80_preEE[(GJet_PT_20to40_MGG_80_preEE.n_electrons + GJet_PT_20to40_MGG_80_preEE.n_muons) == 1]
GJet_PT_40_MGG_80_preEE      = GJet_PT_40_MGG_80_preEE[(GJet_PT_40_MGG_80_preEE.n_electrons + GJet_PT_40_MGG_80_preEE.n_muons) == 1]
# ------------------------------------------------------------------------------------------------------------------------------
GJet_PT_20_MGG_40to80_postEE = GJet_PT_20_MGG_40to80_postEE[(GJet_PT_20_MGG_40to80_postEE.n_electrons + GJet_PT_20_MGG_40to80_postEE.n_muons) == 1]
GJet_PT_20to40_MGG_80_postEE = GJet_PT_20to40_MGG_80_postEE[(GJet_PT_20to40_MGG_80_postEE.n_electrons + GJet_PT_20to40_MGG_80_postEE.n_muons) == 1]
GJet_PT_40_MGG_80_postEE     = GJet_PT_40_MGG_80_postEE[(GJet_PT_40_MGG_80_postEE.n_electrons + GJet_PT_40_MGG_80_postEE.n_muons) == 1]
#########################
########## QCD ##########
#########################
QCD_PT_30to40_MGG_80toInf_preEE   = QCD_PT_30to40_MGG_80toInf_preEE[(QCD_PT_30to40_MGG_80toInf_preEE.n_electrons + QCD_PT_30to40_MGG_80toInf_preEE.n_muons) == 1]
QCD_PT_40toInf_MGG_80toInf_preEE  = QCD_PT_40toInf_MGG_80toInf_preEE[(QCD_PT_40toInf_MGG_80toInf_preEE.n_electrons + QCD_PT_40toInf_MGG_80toInf_preEE.n_muons) == 1]
# ------------------------------------------------------------------------------------------------------------------------------
QCD_PT_30to40_MGG_80toInf_postEE  = QCD_PT_30to40_MGG_80toInf_postEE[(QCD_PT_30to40_MGG_80toInf_postEE.n_electrons + QCD_PT_30to40_MGG_80toInf_postEE.n_muons) == 1]
QCD_PT_40toInf_MGG_80toInf_postEE = QCD_PT_40toInf_MGG_80toInf_postEE[(QCD_PT_40toInf_MGG_80toInf_postEE.n_electrons + QCD_PT_40toInf_MGG_80toInf_postEE.n_muons) == 1]
#########################
########## Top ##########
#########################
TGJets_preEE                  = TGJets_preEE[(TGJets_preEE.n_electrons + TGJets_preEE.n_muons) == 1]
TTG_1Jets_PTG_10to100_preEE   = TTG_1Jets_PTG_10to100_preEE[(TTG_1Jets_PTG_10to100_preEE.n_electrons + TTG_1Jets_PTG_10to100_preEE.n_muons) == 1]
TTG_1Jets_PTG_100to200_preEE  = TTG_1Jets_PTG_100to200_preEE[(TTG_1Jets_PTG_100to200_preEE.n_electrons + TTG_1Jets_PTG_100to200_preEE.n_muons) == 1]
TTG_1Jets_PTG_200_preEE       = TTG_1Jets_PTG_200_preEE[(TTG_1Jets_PTG_200_preEE.n_electrons + TTG_1Jets_PTG_200_preEE.n_muons) == 1]
TTGG_0Jets_preEE              = TTGG_0Jets_preEE[(TTGG_0Jets_preEE.n_electrons + TTGG_0Jets_preEE.n_muons) == 1]
TTto2L2Nu_2Jets_preEE         = TTto2L2Nu_2Jets_preEE[(TTto2L2Nu_2Jets_preEE.n_electrons + TTto2L2Nu_2Jets_preEE.n_muons) == 1]
TTto4Q_2Jets_preEE            = TTto4Q_2Jets_preEE[(TTto4Q_2Jets_preEE.n_electrons + TTto4Q_2Jets_preEE.n_muons) == 1]
TTtoLminusNu2Q_2Jets_preEE    = TTtoLminusNu2Q_2Jets_preEE[(TTtoLminusNu2Q_2Jets_preEE.n_electrons + TTtoLminusNu2Q_2Jets_preEE.n_muons) == 1]
TTtoLplusNu2Q_2Jets_preEE     = TTtoLplusNu2Q_2Jets_preEE[(TTtoLplusNu2Q_2Jets_preEE.n_electrons + TTtoLplusNu2Q_2Jets_preEE.n_muons) == 1]
# ------------------------------------------------------------------------------------------------------------------------------
TGJets_postEE                 = TGJets_postEE[(TGJets_postEE.n_electrons + TGJets_postEE.n_muons) == 1]
TTG_1Jets_PTG_10to100_postEE  = TTG_1Jets_PTG_10to100_postEE[(TTG_1Jets_PTG_10to100_postEE.n_electrons + TTG_1Jets_PTG_10to100_postEE.n_muons) == 1]
TTG_1Jets_PTG_100to200_postEE = TTG_1Jets_PTG_100to200_postEE[(TTG_1Jets_PTG_100to200_postEE.n_electrons + TTG_1Jets_PTG_100to200_postEE.n_muons) == 1]
TTG_1Jets_PTG_200_postEE      = TTG_1Jets_PTG_200_postEE[(TTG_1Jets_PTG_200_postEE.n_electrons + TTG_1Jets_PTG_200_postEE.n_muons) == 1]
TTGG_0Jets_postEE             = TTGG_0Jets_postEE[(TTGG_0Jets_postEE.n_electrons + TTGG_0Jets_postEE.n_muons) == 1]
TTto2L2Nu_2Jets_postEE        = TTto2L2Nu_2Jets_postEE[(TTto2L2Nu_2Jets_postEE.n_electrons + TTto2L2Nu_2Jets_postEE.n_muons) == 1]
TTto4Q_2Jets_postEE           = TTto4Q_2Jets_postEE[(TTto4Q_2Jets_postEE.n_electrons + TTto4Q_2Jets_postEE.n_muons) == 1]
TTtoLminusNu2Q_2Jets_postEE   = TTtoLminusNu2Q_2Jets_postEE[(TTtoLminusNu2Q_2Jets_postEE.n_electrons + TTtoLminusNu2Q_2Jets_postEE.n_muons) == 1]
TTtoLplusNu2Q_2Jets_postEE    = TTtoLplusNu2Q_2Jets_postEE[(TTtoLplusNu2Q_2Jets_postEE.n_electrons + TTtoLplusNu2Q_2Jets_postEE.n_muons) == 1]
########################
########## WG ##########
########################
WGtoLNuG_1Jets_PTG_10to100_preEE   = WGtoLNuG_1Jets_PTG_10to100_preEE[(WGtoLNuG_1Jets_PTG_10to100_preEE.n_electrons + WGtoLNuG_1Jets_PTG_10to100_preEE.n_muons) == 1]
WGtoLNuG_1Jets_PTG_100to200_preEE  = WGtoLNuG_1Jets_PTG_100to200_preEE[(WGtoLNuG_1Jets_PTG_100to200_preEE.n_electrons + WGtoLNuG_1Jets_PTG_100to200_preEE.n_muons) == 1]
WGtoLNuG_1Jets_PTG_200to400_preEE  = WGtoLNuG_1Jets_PTG_200to400_preEE[(WGtoLNuG_1Jets_PTG_200to400_preEE.n_electrons + WGtoLNuG_1Jets_PTG_200to400_preEE.n_muons) == 1]
WGtoLNuG_1Jets_PTG_400to600_preEE  = WGtoLNuG_1Jets_PTG_400to600_preEE[(WGtoLNuG_1Jets_PTG_400to600_preEE.n_electrons + WGtoLNuG_1Jets_PTG_400to600_preEE.n_muons) == 1]
WGtoLNuG_1Jets_PTG_600_preEE       = WGtoLNuG_1Jets_PTG_600_preEE[(WGtoLNuG_1Jets_PTG_600_preEE.n_electrons + WGtoLNuG_1Jets_PTG_600_preEE.n_muons) == 1]
# ------------------------------------------------------------------------------------------------------------------------------
WGtoLNuG_1Jets_PTG_10to100_postEE  = WGtoLNuG_1Jets_PTG_10to100_postEE[(WGtoLNuG_1Jets_PTG_10to100_postEE.n_electrons + WGtoLNuG_1Jets_PTG_10to100_postEE.n_muons) == 1]
WGtoLNuG_1Jets_PTG_100to200_postEE = WGtoLNuG_1Jets_PTG_100to200_postEE[(WGtoLNuG_1Jets_PTG_100to200_postEE.n_electrons + WGtoLNuG_1Jets_PTG_100to200_postEE.n_muons) == 1]
WGtoLNuG_1Jets_PTG_200to400_postEE = WGtoLNuG_1Jets_PTG_200to400_postEE[(WGtoLNuG_1Jets_PTG_200to400_postEE.n_electrons + WGtoLNuG_1Jets_PTG_200to400_postEE.n_muons) == 1]
WGtoLNuG_1Jets_PTG_400to600_postEE = WGtoLNuG_1Jets_PTG_400to600_postEE[(WGtoLNuG_1Jets_PTG_400to600_postEE.n_electrons + WGtoLNuG_1Jets_PTG_400to600_postEE.n_muons) == 1]
WGtoLNuG_1Jets_PTG_600_postEE      = WGtoLNuG_1Jets_PTG_600_postEE[(WGtoLNuG_1Jets_PTG_600_postEE.n_electrons + WGtoLNuG_1Jets_PTG_600_postEE.n_muons) == 1]
########################
########## ZG ##########
########################
ZGto2LG_1Jets_preEE                = ZGto2LG_1Jets_preEE[(ZGto2LG_1Jets_preEE.n_electrons + ZGto2LG_1Jets_preEE.n_muons) == 1]
ZGto2NuG_1Jets_PTG_10to100_preEE   = ZGto2NuG_1Jets_PTG_10to100_preEE[(ZGto2NuG_1Jets_PTG_10to100_preEE.n_electrons + ZGto2NuG_1Jets_PTG_10to100_preEE.n_muons) == 1]
ZGto2NuG_1Jets_PTG_100to200_preEE  = ZGto2NuG_1Jets_PTG_100to200_preEE[(ZGto2NuG_1Jets_PTG_100to200_preEE.n_electrons + ZGto2NuG_1Jets_PTG_100to200_preEE.n_muons) == 1]
ZGto2NuG_1Jets_PTG_200to400_preEE  = ZGto2NuG_1Jets_PTG_200to400_preEE[(ZGto2NuG_1Jets_PTG_200to400_preEE.n_electrons + ZGto2NuG_1Jets_PTG_200to400_preEE.n_muons) == 1]
ZGto2NuG_1Jets_PTG_400to600_preEE  = ZGto2NuG_1Jets_PTG_400to600_preEE[(ZGto2NuG_1Jets_PTG_400to600_preEE.n_electrons + ZGto2NuG_1Jets_PTG_400to600_preEE.n_muons) == 1]
ZGto2NuG_1Jets_PTG_600_preEE       = ZGto2NuG_1Jets_PTG_600_preEE[(ZGto2NuG_1Jets_PTG_600_preEE.n_electrons + ZGto2NuG_1Jets_PTG_600_preEE.n_muons) == 1]
# ------------------------------------------------------------------------------------------------------------------------------
ZGto2LG_1Jets_postEE               = ZGto2LG_1Jets_postEE[(ZGto2LG_1Jets_postEE.n_electrons + ZGto2LG_1Jets_postEE.n_muons) == 1]
ZGto2NuG_1Jets_PTG_10to100_postEE  = ZGto2NuG_1Jets_PTG_10to100_postEE[(ZGto2NuG_1Jets_PTG_10to100_postEE.n_electrons + ZGto2NuG_1Jets_PTG_10to100_postEE.n_muons) == 1]
ZGto2NuG_1Jets_PTG_100to200_postEE = ZGto2NuG_1Jets_PTG_100to200_postEE[(ZGto2NuG_1Jets_PTG_100to200_postEE.n_electrons + ZGto2NuG_1Jets_PTG_100to200_postEE.n_muons) == 1]
ZGto2NuG_1Jets_PTG_200to400_postEE = ZGto2NuG_1Jets_PTG_200to400_postEE[(ZGto2NuG_1Jets_PTG_200to400_postEE.n_electrons + ZGto2NuG_1Jets_PTG_200to400_postEE.n_muons) == 1]
ZGto2NuG_1Jets_PTG_400to600_postEE = ZGto2NuG_1Jets_PTG_400to600_postEE[(ZGto2NuG_1Jets_PTG_400to600_postEE.n_electrons + ZGto2NuG_1Jets_PTG_400to600_postEE.n_muons) == 1]
ZGto2NuG_1Jets_PTG_600_postEE      = ZGto2NuG_1Jets_PTG_600_postEE[(ZGto2NuG_1Jets_PTG_600_postEE.n_electrons + ZGto2NuG_1Jets_PTG_600_postEE.n_muons) == 1]
###############################
########## WH-signal ##########
###############################
WminusH_Hto2G_WtoLNu_M_125_preEE  = WminusH_Hto2G_WtoLNu_M_125_preEE[(WminusH_Hto2G_WtoLNu_M_125_preEE.n_electrons + WminusH_Hto2G_WtoLNu_M_125_preEE.n_muons) == 1]
WplusH_Hto2G_WtoLNu_M_125_preEE   = WplusH_Hto2G_WtoLNu_M_125_preEE[(WplusH_Hto2G_WtoLNu_M_125_preEE.n_electrons + WplusH_Hto2G_WtoLNu_M_125_preEE.n_muons) == 1]
# ------------------------------------------------------------------------------------------------------------------------------
WminusH_Hto2G_WtoLNu_M_125_postEE = WminusH_Hto2G_WtoLNu_M_125_postEE[(WminusH_Hto2G_WtoLNu_M_125_postEE.n_electrons + WminusH_Hto2G_WtoLNu_M_125_postEE.n_muons) == 1]
WplusH_Hto2G_WtoLNu_M_125_postEE  = WplusH_Hto2G_WtoLNu_M_125_postEE[(WplusH_Hto2G_WtoLNu_M_125_postEE.n_electrons + WplusH_Hto2G_WtoLNu_M_125_postEE.n_muons) == 1]
################################################################################################################################
########## Calculate the weights of each MC sample. ##########
################################################################################################################################
# Luminosity ~ 1/fb, cross-section ~ pb = 1,000 fb.
preEE_luminosity  = 7.9804 * 1000
postEE_luminosity = 26.6717 * 1000
########################
########## VV ##########
########################
WW_preEE_weight  = preEE_luminosity * 80.23 * (WW_preEE.genWeight / 15405620.140625)
WZ_preEE_weight  = preEE_luminosity * 29.1 * (WZ_preEE.genWeight / 7479528.0)
ZZ_preEE_weight  = preEE_luminosity * 12.75 * (ZZ_preEE.genWeight / 1181750.0)
# ------------------------------------------------------------------------------------------------------------------------------
WW_postEE_weight = postEE_luminosity * 80.23 * (WW_postEE.genWeight / 53112474.8125)
WZ_postEE_weight = postEE_luminosity * 29.1 * (WZ_postEE.genWeight / 26722782.0)
ZZ_postEE_weight = postEE_luminosity * 12.75 * (ZZ_postEE.genWeight / 4043040.0)
###########################################
########## Diphoton-nonresonance ##########
###########################################
Diphoton_preEE_weight  = preEE_luminosity * 88.75 * (Diphoton_preEE.genWeight / 6852451.23828125)
# ------------------------------------------------------------------------------------------------------------------------------
Diphoton_postEE_weight = postEE_luminosity * 88.75 * (Diphoton_postEE.genWeight / 21291656.96875)
############################
########## DYJets ##########
############################
DYto2L_2Jets_MLL_50_preEE_weight                 = preEE_luminosity * 6688.0 * (DYto2L_2Jets_MLL_50_preEE.genWeight / 1211348605824.0)
DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE_weight    = preEE_luminosity * 124 * (DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE.genWeight / 6787147744.0)
DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE_weight   = preEE_luminosity * 2.088 * (DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE.genWeight / 1103956653.0)
DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE_weight  = preEE_luminosity * 0.3493 * (DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE.genWeight / 401854.78125)
DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE_weight  = preEE_luminosity * 0.04331 * (DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE.genWeight / 75190.2919921875)
DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE_weight  = preEE_luminosity * 0.00313 * (DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE.genWeight / 4245.39599609375)
DYGto2LG_1Jets_MLL_50_PTG_600_preEE_weight       = preEE_luminosity * 0.0006528 * (DYGto2LG_1Jets_MLL_50_PTG_600_preEE.genWeight / 728.2471923828125)
# ------------------------------------------------------------------------------------------------------------------------------
DYto2L_2Jets_MLL_50_postEE_weight                = postEE_luminosity * 6688.0 * (DYto2L_2Jets_MLL_50_postEE.genWeight / 2449564418560.0)
DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE_weight   = postEE_luminosity * 124 * (DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE.genWeight / 23980987752.0)
DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE_weight  = postEE_luminosity * 2.088 * (DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE.genWeight / 3347917097.5)
DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE_weight = postEE_luminosity * 0.3493 * (DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE.genWeight / 1393253.8125)
DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE_weight = postEE_luminosity * 0.04331 * (DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE.genWeight / 264963.9267578125)
DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE_weight = postEE_luminosity * 0.00313 * (DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE.genWeight / 15294.290283203125)
DYGto2LG_1Jets_MLL_50_PTG_600_postEE_weight      = postEE_luminosity * 0.0006528 * (DYGto2LG_1Jets_MLL_50_PTG_600_postEE.genWeight / 2667.3897132873535)
###########################
########## GJets ##########
###########################
GJet_PT_20_MGG_40to80_preEE_weight = preEE_luminosity * 3284.0 * (GJet_PT_20_MGG_40to80_preEE.genWeight / 1572670.0)
GJet_PT_20to40_MGG_80_preEE_weight = preEE_luminosity * 242.5 * (GJet_PT_20to40_MGG_80_preEE.genWeight / 6765159.0)
GJet_PT_40_MGG_80_preEE_weight     = preEE_luminosity * 919.1 * (GJet_PT_40_MGG_80_preEE.genWeight / 5009479.0) 
# ------------------------------------------------------------------------------------------------------------------------------
GJet_PT_20_MGG_40to80_postEE_weight = postEE_luminosity * 3284.0 * (GJet_PT_20_MGG_40to80_postEE.genWeight / 11040768.0)
GJet_PT_20to40_MGG_80_postEE_weight = postEE_luminosity * 242.5 * (GJet_PT_20to40_MGG_80_postEE.genWeight / 10016843.0)
GJet_PT_40_MGG_80_postEE_weight     = postEE_luminosity * 919.1 * (GJet_PT_40_MGG_80_postEE.genWeight / 19884039.0)
#########################
########## QCD ##########
#########################
QCD_PT_30to40_MGG_80toInf_preEE_weight   = preEE_luminosity * 25950.0 * (QCD_PT_30to40_MGG_80toInf_preEE.genWeight / 1420141.0)
QCD_PT_40toInf_MGG_80toInf_preEE_weight  = preEE_luminosity * 124700.0 * (QCD_PT_40toInf_MGG_80toInf_preEE.genWeight / 2947774.0)
# ------------------------------------------------------------------------------------------------------------------------------
QCD_PT_30to40_MGG_80toInf_postEE_weight  = postEE_luminosity * 25950.0 * (QCD_PT_30to40_MGG_80toInf_postEE.genWeight / 4496412.0)
QCD_PT_40toInf_MGG_80toInf_postEE_weight = postEE_luminosity * 124700.0 * (QCD_PT_40toInf_MGG_80toInf_postEE.genWeight / 10256522.0)
#########################
########## Top ##########
#########################
TGJets_preEE_weight                  = preEE_luminosity * 3.873 * (TGJets_preEE.genWeight / 8590937.5)
TTG_1Jets_PTG_10to100_preEE_weight   = preEE_luminosity * 4.216 * (TTG_1Jets_PTG_10to100_preEE.genWeight / 7786355.0)
TTG_1Jets_PTG_100to200_preEE_weight  = preEE_luminosity * 0.4114 * (TTG_1Jets_PTG_100to200_preEE.genWeight / 492546.3046875)
TTG_1Jets_PTG_200_preEE_weight       = preEE_luminosity * 0.1284 * (TTG_1Jets_PTG_200_preEE.genWeight / 105009.28125)
TTGG_0Jets_preEE_weight              = preEE_luminosity * 0.02391 * (TTGG_0Jets_preEE.genWeight / 16453.898612976074)
TTto2L2Nu_2Jets_preEE_weight         = preEE_luminosity * 95.49 * (TTto2L2Nu_2Jets_preEE.genWeight / 11157318184.0)
TTto4Q_2Jets_preEE_weight            = preEE_luminosity * 385.5 * (TTto4Q_2Jets_preEE.genWeight / 78961823008.0)
TTtoLminusNu2Q_2Jets_preEE_weight    = preEE_luminosity * 190.4 * (TTtoLminusNu2Q_2Jets_preEE.genWeight / 28002971856.0)
TTtoLplusNu2Q_2Jets_preEE_weight     = preEE_luminosity * 192.6 * (TTtoLplusNu2Q_2Jets_preEE.genWeight / 31175028800.0)
# ------------------------------------------------------------------------------------------------------------------------------
TGJets_postEE_weight                 = postEE_luminosity * 3.873 * (TGJets_postEE.genWeight / 29158635.1875)
TTG_1Jets_PTG_10to100_postEE_weight  = postEE_luminosity * 4.216 * (TTG_1Jets_PTG_10to100_postEE.genWeight / 28230677.25)
TTG_1Jets_PTG_100to200_postEE_weight = postEE_luminosity * 0.4114 * (TTG_1Jets_PTG_100to200_postEE.genWeight / 1539109.1171875)
TTG_1Jets_PTG_200_postEE_weight      = postEE_luminosity * 0.1284 * (TTG_1Jets_PTG_200_postEE.genWeight / 383827.5)
TTGG_0Jets_postEE_weight             = postEE_luminosity * 0.02391 * (TTGG_0Jets_postEE.genWeight / 58773.11389160156)
TTto2L2Nu_2Jets_postEE_weight        = postEE_luminosity * 95.49 * (TTto2L2Nu_2Jets_postEE.genWeight / 39761414604.0)
TTto4Q_2Jets_postEE_weight           = postEE_luminosity * 385.5 * (TTto4Q_2Jets_postEE.genWeight / 291610346048.0)
TTtoLminusNu2Q_2Jets_postEE_weight   = postEE_luminosity * 190.4 * (TTtoLminusNu2Q_2Jets_postEE.genWeight / 104466959720.0)
TTtoLplusNu2Q_2Jets_postEE_weight    = postEE_luminosity * 192.6 * (TTtoLplusNu2Q_2Jets_postEE.genWeight / 106209821120.0)
########################
########## WG ##########
########################
WGtoLNuG_1Jets_PTG_10to100_preEE_weight   = preEE_luminosity * 662.2 * (WGtoLNuG_1Jets_PTG_10to100_preEE.genWeight / 64148028416.0)
WGtoLNuG_1Jets_PTG_100to200_preEE_weight  = preEE_luminosity * 2.221 * (WGtoLNuG_1Jets_PTG_100to200_preEE.genWeight / 11610085.3125)
WGtoLNuG_1Jets_PTG_200to400_preEE_weight  = preEE_luminosity * 0.2908 * (WGtoLNuG_1Jets_PTG_200to400_preEE.genWeight / 425685.3125)
WGtoLNuG_1Jets_PTG_400to600_preEE_weight  = preEE_luminosity * 0.02231 * (WGtoLNuG_1Jets_PTG_400to600_preEE.genWeight / 24862.81689453125)
WGtoLNuG_1Jets_PTG_600_preEE_weight       = preEE_luminosity * 0.004907 * (WGtoLNuG_1Jets_PTG_600_preEE.genWeight / 4764.3125)
# ------------------------------------------------------------------------------------------------------------------------------
WGtoLNuG_1Jets_PTG_10to100_postEE_weight  = postEE_luminosity * 662.2 * (WGtoLNuG_1Jets_PTG_10to100_postEE.genWeight / 213809037344.0)
WGtoLNuG_1Jets_PTG_100to200_postEE_weight = postEE_luminosity * 2.221 * (WGtoLNuG_1Jets_PTG_100to200_postEE.genWeight / 41071357.0625)
WGtoLNuG_1Jets_PTG_200to400_postEE_weight = postEE_luminosity * 0.2908 * (WGtoLNuG_1Jets_PTG_200to400_postEE.genWeight / 1441224.4375)
WGtoLNuG_1Jets_PTG_400to600_postEE_weight = postEE_luminosity * 0.02231 * (WGtoLNuG_1Jets_PTG_400to600_postEE.genWeight / 86582.134765625)
WGtoLNuG_1Jets_PTG_600_postEE_weight      = postEE_luminosity * 0.004907 * (WGtoLNuG_1Jets_PTG_600_postEE.genWeight / 17752.469146728516)
########################
########## ZG ##########
########################
ZGto2NuG_1Jets_PTG_10to100_preEE_weight   = preEE_luminosity * 39.93 * (ZGto2NuG_1Jets_PTG_10to100_preEE.genWeight / 683170892.0)
ZGto2NuG_1Jets_PTG_100to200_preEE_weight  = preEE_luminosity * 0.5652 * (ZGto2NuG_1Jets_PTG_100to200_preEE.genWeight / 1542443.828125)
ZGto2NuG_1Jets_PTG_200to400_preEE_weight  = preEE_luminosity * 0.07535 * (ZGto2NuG_1Jets_PTG_200to400_preEE.genWeight / 130343.69342041016)
ZGto2NuG_1Jets_PTG_400to600_preEE_weight  = preEE_luminosity * 0.005547 * (ZGto2NuG_1Jets_PTG_400to600_preEE.genWeight / 7756.102783203125)
ZGto2NuG_1Jets_PTG_600_preEE_weight       = preEE_luminosity * 0.001177 * (ZGto2NuG_1Jets_PTG_600_preEE.genWeight / 1267.0567321777344)
ZGto2LG_1Jets_preEE_weight                = preEE_luminosity * 1.075 * (ZGto2LG_1Jets_preEE.genWeight / 937276.0)
# ------------------------------------------------------------------------------------------------------------------------------
ZGto2NuG_1Jets_PTG_10to100_postEE_weight  = postEE_luminosity * 39.93 * (ZGto2NuG_1Jets_PTG_10to100_postEE.genWeight / 2224161750.0)
ZGto2NuG_1Jets_PTG_100to200_postEE_weight = postEE_luminosity * 0.5652 * (ZGto2NuG_1Jets_PTG_100to200_postEE.genWeight / 5272298.0625)
ZGto2NuG_1Jets_PTG_200to400_postEE_weight = postEE_luminosity * 0.07535 * (ZGto2NuG_1Jets_PTG_200to400_postEE.genWeight / 481404.134765625)
ZGto2NuG_1Jets_PTG_400to600_postEE_weight = postEE_luminosity * 0.005547 * (ZGto2NuG_1Jets_PTG_400to600_postEE.genWeight / 25661.05078125)
ZGto2NuG_1Jets_PTG_600_postEE_weight      = postEE_luminosity * 0.001177 * (ZGto2NuG_1Jets_PTG_600_postEE.genWeight / 4735.676239013672)
ZGto2LG_1Jets_postEE_weight               = postEE_luminosity * 1.075 * (ZGto2LG_1Jets_postEE.genWeight / 2910558.0)
###############################
########## WH-signal ##########
###############################
WminusH_Hto2G_WtoLNu_M_125_preEE_weight  = preEE_luminosity * 0.1888 * 0.0025 * (WminusH_Hto2G_WtoLNu_M_125_preEE.genWeight / 10294.633354187012)
WplusH_Hto2G_WtoLNu_M_125_preEE_weight   = preEE_luminosity * 0.2999 * 0.0025 * (WplusH_Hto2G_WtoLNu_M_125_preEE.genWeight / 16319.952537536621)
# ------------------------------------------------------------------------------------------------------------------------------
WminusH_Hto2G_WtoLNu_M_125_postEE_weight = postEE_luminosity * 0.1888 * 0.0025 * (WminusH_Hto2G_WtoLNu_M_125_postEE.genWeight / 37705.549980163574)
WplusH_Hto2G_WtoLNu_M_125_postEE_weight  = postEE_luminosity * 0.2999 * 0.0025 * (WplusH_Hto2G_WtoLNu_M_125_postEE.genWeight / 57964.71792221069)
################################################################################################################################
########## Concatenate 2022 preEE and postEE Data and MC NTuples of the same process ##########
################################################################################################################################
Data = np.concatenate([DataC_preEE, DataD_preEE, DataE_postEE, DataF_postEE, DataG_postEE])
Diphoton = np.concatenate([Diphoton_preEE, Diphoton_postEE])
DYJets = np.concatenate(
    [
        DYto2L_2Jets_MLL_50_preEE, DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE, DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE, DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE,
        DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE, DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE, DYGto2LG_1Jets_MLL_50_PTG_600_preEE, 
        DYto2L_2Jets_MLL_50_postEE, DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE, DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE, DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE,
        DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE, DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE, DYGto2LG_1Jets_MLL_50_PTG_600_postEE
    ]
)
GJets = np.concatenate(
    [
        GJet_PT_20_MGG_40to80_preEE, GJet_PT_20to40_MGG_80_preEE, GJet_PT_40_MGG_80_preEE,
        GJet_PT_20_MGG_40to80_postEE, GJet_PT_20to40_MGG_80_postEE, GJet_PT_40_MGG_80_postEE
    ]
)
QCD = np.concatenate(
    [
        QCD_PT_30to40_MGG_80toInf_preEE, QCD_PT_40toInf_MGG_80toInf_preEE,
        QCD_PT_30to40_MGG_80toInf_postEE, QCD_PT_40toInf_MGG_80toInf_postEE
    ]
)
Top = np.concatenate(
    [
        TGJets_preEE, TTG_1Jets_PTG_10to100_preEE, TTG_1Jets_PTG_100to200_preEE, TTG_1Jets_PTG_200_preEE, TTGG_0Jets_preEE, 
        TTto2L2Nu_2Jets_preEE, TTto4Q_2Jets_preEE, TTtoLminusNu2Q_2Jets_preEE, TTtoLplusNu2Q_2Jets_preEE,
        TGJets_postEE, TTG_1Jets_PTG_10to100_postEE, TTG_1Jets_PTG_100to200_postEE, TTG_1Jets_PTG_200_postEE, TTGG_0Jets_postEE, 
        TTto2L2Nu_2Jets_postEE, TTto4Q_2Jets_postEE, TTtoLminusNu2Q_2Jets_postEE, TTtoLplusNu2Q_2Jets_postEE
    ]
)
Diboson = np.concatenate(
    [
        WW_preEE, WZ_preEE, ZZ_preEE, WW_postEE, WZ_postEE, ZZ_postEE,
        WGtoLNuG_1Jets_PTG_10to100_preEE, WGtoLNuG_1Jets_PTG_100to200_preEE, WGtoLNuG_1Jets_PTG_200to400_preEE, WGtoLNuG_1Jets_PTG_400to600_preEE, WGtoLNuG_1Jets_PTG_600_preEE, WGtoLNuG_1Jets_PTG_10to100_postEE, WGtoLNuG_1Jets_PTG_100to200_postEE, WGtoLNuG_1Jets_PTG_200to400_postEE, WGtoLNuG_1Jets_PTG_400to600_postEE, WGtoLNuG_1Jets_PTG_600_postEE,
        ZGto2LG_1Jets_preEE, ZGto2NuG_1Jets_PTG_10to100_preEE, ZGto2NuG_1Jets_PTG_100to200_preEE, ZGto2NuG_1Jets_PTG_200to400_preEE, ZGto2NuG_1Jets_PTG_400to600_preEE, ZGto2NuG_1Jets_PTG_600_preEE, ZGto2LG_1Jets_postEE, ZGto2NuG_1Jets_PTG_10to100_postEE, ZGto2NuG_1Jets_PTG_100to200_postEE, ZGto2NuG_1Jets_PTG_200to400_postEE, ZGto2NuG_1Jets_PTG_400to600_postEE, ZGto2NuG_1Jets_PTG_600_postEE
    ]
)
WH_signal = np.concatenate(
    [
        WminusH_Hto2G_WtoLNu_M_125_preEE, WplusH_Hto2G_WtoLNu_M_125_preEE,
        WminusH_Hto2G_WtoLNu_M_125_postEE, WplusH_Hto2G_WtoLNu_M_125_postEE
    ]
)
################################################################################################################################
########## Max and Min gamma mvaID ##########
################################################################################################################################
Data_max_gamma_ID      = np.maximum(Data.lead_mvaID, Data.sublead_mvaID)
Data_min_gamma_ID      = np.minimum(Data.lead_mvaID, Data.sublead_mvaID)
Diphoton_max_gamma_ID  = np.maximum(Diphoton.lead_mvaID, Diphoton.sublead_mvaID)
Diphoton_min_gamma_ID  = np.minimum(Diphoton.lead_mvaID, Diphoton.sublead_mvaID)
DYJets_max_gamma_ID    = np.maximum(DYJets.lead_mvaID, DYJets.sublead_mvaID)
DYJets_min_gamma_ID    = np.minimum(DYJets.lead_mvaID, DYJets.sublead_mvaID)
GJets_max_gamma_ID     = np.maximum(GJets.lead_mvaID, GJets.sublead_mvaID)
GJets_min_gamma_ID     = np.minimum(GJets.lead_mvaID, GJets.sublead_mvaID)
QCD_max_gamma_ID       = np.maximum(QCD.lead_mvaID, QCD.sublead_mvaID)
QCD_min_gamma_ID       = np.minimum(QCD.lead_mvaID, QCD.sublead_mvaID)
Top_max_gamma_ID       = np.maximum(Top.lead_mvaID, Top.sublead_mvaID)
Top_min_gamma_ID       = np.minimum(Top.lead_mvaID, Top.sublead_mvaID)
Diboson_max_gamma_ID   = np.maximum(Diboson.lead_mvaID, Diboson.sublead_mvaID)
Diboson_min_gamma_ID   = np.minimum(Diboson.lead_mvaID, Diboson.sublead_mvaID)
WH_signal_max_gamma_ID = np.maximum(WH_signal.lead_mvaID, WH_signal.sublead_mvaID)
WH_signal_min_gamma_ID = np.minimum(WH_signal.lead_mvaID, WH_signal.sublead_mvaID)
################################################################################################################################
########## Concatenate the MC weights of the same process ##########
################################################################################################################################
Diphoton_weight = np.concatenate([Diphoton_preEE_weight, Diphoton_postEE_weight])
DYJets_weight = np.concatenate(
    [
        DYto2L_2Jets_MLL_50_preEE_weight, DYGto2LG_1Jets_MLL_50_PTG_10to50_preEE_weight, DYGto2LG_1Jets_MLL_50_PTG_50to100_preEE_weight, DYGto2LG_1Jets_MLL_50_PTG_100to200_preEE_weight,
        DYGto2LG_1Jets_MLL_50_PTG_200to400_preEE_weight, DYGto2LG_1Jets_MLL_50_PTG_400to600_preEE_weight, DYGto2LG_1Jets_MLL_50_PTG_600_preEE_weight,
        DYto2L_2Jets_MLL_50_postEE_weight, DYGto2LG_1Jets_MLL_50_PTG_10to50_postEE_weight, DYGto2LG_1Jets_MLL_50_PTG_50to100_postEE_weight, DYGto2LG_1Jets_MLL_50_PTG_100to200_postEE_weight,
        DYGto2LG_1Jets_MLL_50_PTG_200to400_postEE_weight, DYGto2LG_1Jets_MLL_50_PTG_400to600_postEE_weight, DYGto2LG_1Jets_MLL_50_PTG_600_postEE_weight
    ]
)
GJets_weight = np.concatenate(
    [
        GJet_PT_20_MGG_40to80_preEE_weight, GJet_PT_20to40_MGG_80_preEE_weight, GJet_PT_40_MGG_80_preEE_weight,
        GJet_PT_20_MGG_40to80_postEE_weight, GJet_PT_20to40_MGG_80_postEE_weight, GJet_PT_40_MGG_80_postEE_weight
    ]
)
QCD_weight = np.concatenate(
    [
        QCD_PT_30to40_MGG_80toInf_preEE_weight, QCD_PT_40toInf_MGG_80toInf_preEE_weight,
        QCD_PT_30to40_MGG_80toInf_postEE_weight, QCD_PT_40toInf_MGG_80toInf_postEE_weight
    ]
)
Top_weight = np.concatenate(
    [
        TGJets_preEE_weight, TTG_1Jets_PTG_10to100_preEE_weight, TTG_1Jets_PTG_100to200_preEE_weight, TTG_1Jets_PTG_200_preEE_weight, TTGG_0Jets_preEE_weight, 
        TTto2L2Nu_2Jets_preEE_weight, TTto4Q_2Jets_preEE_weight, TTtoLminusNu2Q_2Jets_preEE_weight, TTtoLplusNu2Q_2Jets_preEE_weight,
        TGJets_postEE_weight, TTG_1Jets_PTG_10to100_postEE_weight, TTG_1Jets_PTG_100to200_postEE_weight, TTG_1Jets_PTG_200_postEE_weight, TTGG_0Jets_postEE_weight,
        TTto2L2Nu_2Jets_postEE_weight, TTto4Q_2Jets_postEE_weight, TTtoLminusNu2Q_2Jets_postEE_weight, TTtoLplusNu2Q_2Jets_postEE_weight
    ]
)
Diboson_weight = np.concatenate(
    [
        WW_preEE_weight, WZ_preEE_weight, ZZ_preEE_weight, WW_postEE_weight, WZ_postEE_weight, ZZ_postEE_weight,
        WGtoLNuG_1Jets_PTG_10to100_preEE_weight, WGtoLNuG_1Jets_PTG_100to200_preEE_weight, WGtoLNuG_1Jets_PTG_200to400_preEE_weight, WGtoLNuG_1Jets_PTG_400to600_preEE_weight, WGtoLNuG_1Jets_PTG_600_preEE_weight, WGtoLNuG_1Jets_PTG_10to100_postEE_weight, WGtoLNuG_1Jets_PTG_100to200_postEE_weight, WGtoLNuG_1Jets_PTG_200to400_postEE_weight, WGtoLNuG_1Jets_PTG_400to600_postEE_weight, WGtoLNuG_1Jets_PTG_600_postEE_weight,
        ZGto2LG_1Jets_preEE_weight, ZGto2NuG_1Jets_PTG_10to100_preEE_weight, ZGto2NuG_1Jets_PTG_100to200_preEE_weight, ZGto2NuG_1Jets_PTG_200to400_preEE_weight, ZGto2NuG_1Jets_PTG_400to600_preEE_weight, ZGto2NuG_1Jets_PTG_600_preEE_weight, 
        ZGto2LG_1Jets_postEE_weight, ZGto2NuG_1Jets_PTG_10to100_postEE_weight, ZGto2NuG_1Jets_PTG_100to200_postEE_weight, ZGto2NuG_1Jets_PTG_200to400_postEE_weight, ZGto2NuG_1Jets_PTG_400to600_postEE_weight, ZGto2NuG_1Jets_PTG_600_postEE_weight
    ]
)
WH_signal_weight = np.concatenate(
    [
        WminusH_Hto2G_WtoLNu_M_125_preEE_weight, WplusH_Hto2G_WtoLNu_M_125_preEE_weight,
        WminusH_Hto2G_WtoLNu_M_125_postEE_weight, WplusH_Hto2G_WtoLNu_M_125_postEE_weight
    ]
)
################################################################################################################################
########## Labels for each MC sample ##########
################################################################################################################################
labels = ["Diphoton", "DYJets", "$\it{\gamma}+jets$", "QCD", "Top", "$VV/V+\it{\gamma}$"]
################################################################################################################################
########## Plotting ##########
################################################################################################################################
hep.style.use("CMS")
suffix = "20250624"
#######################################################################################
########## Diphoton mass (With data/MC ratio plot & data in errorbar style.) ##########
#######################################################################################
print("~~~~~~~~~~~[INFO] diphoton mass plot started.~~~~~~~~~~~")
Data_mass = ak.to_numpy(Data.mass)
Data_mass_hist, bins = np.histogram(Data_mass, range = (100, 180), bins = 70)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphoton.mass, DYJets.mass, GJets.mass, QCD.mass, Top.mass, Diboson.mass]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_mass = ak.to_numpy(WH_signal.mass)
WH_signal_mass_hist, _ = np.histogram(
    WH_signal_mass,
    bins = bins,
    weights = WH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex = True, figsize = (10, 12))

# Upper plot: Data and MC comparison
# When you measure the number of events N, each event is independent and has a constant probability per unit time or per measurement. 
# Under these conditions, the probability distribution that describes the observed number of events is the Poisson distribution.
# Since the mean equals to variance in Poisson distribution, the statistical uncertainty is given by the square root of N.
axs[0].errorbar(
    x = bin_center, y = Data_mass_hist,
    yerr = np.sqrt(Data_mass_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_mass, bins = bins, weights = WH_signal_weight * 50,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 50$"
)
axs[0].set_xlim((100, 180))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(ncol = 2, loc = "upper right")

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_mass_hist, MC_hist,
    out = np.zeros_like(Data_mass_hist, dtype=float),
    where = MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_mass_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_mass_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_mass_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_mass_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_mass_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("$m_{\gamma\gamma} \ (GeV)$")

# Save the plot
plt.savefig(f"diphoton_mass_{suffix}.png")
print("~~~~~~~~~~~[INFO] diphoton mass plot completed.~~~~~~~~~~~")
#############################################################################################
########## lead photon pt/mass (With data/MC ratio plot & data in errorbar style.) ##########
#############################################################################################
print("~~~~~~~~~~~[INFO] lead photon pt to diphoton mass plot started.~~~~~~~~~~~")
Data_lead_pt_mass = ak.to_numpy(Data.lead_pt / Data.mass)
Data_lead_pt_mass_hist, bins = np.histogram(Data_lead_pt_mass, range = (0.3, 1.7), bins = 70)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphoton.lead_pt / Diphoton.mass, DYJets.lead_pt / DYJets.mass, GJets.lead_pt / GJets.mass, QCD.lead_pt / QCD.mass, Top.lead_pt / Top.mass, Diboson.lead_pt / Diboson.mass]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_lead_pt_mass = ak.to_numpy(WH_signal.lead_pt / WH_signal.mass)
WH_signal_lead_pt_mass_hist, _ = np.histogram(
    WH_signal_lead_pt_mass,
    bins = bins,
    weights = WH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex = True, figsize = (10, 12))

# Upper plot: Data and MC comparison
# When you measure the number of events N, each event is independent and has a constant probability per unit time or per measurement. 
# Under these conditions, the probability distribution that describes the observed number of events is the Poisson distribution.
# Since the mean equals to variance in Poisson distribution, the statistical uncertainty is given by the square root of N.
axs[0].errorbar(
    x = bin_center, y = Data_lead_pt_mass_hist,
    yerr = np.sqrt(Data_lead_pt_mass_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_lead_pt_mass, bins = bins, weights = WH_signal_weight * 100,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 100$"
)
axs[0].set_xlim((0.3, 1.7))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend()

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_lead_pt_mass_hist, MC_hist,
    out = np.zeros_like(Data_lead_pt_mass_hist, dtype=float),
    where = MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_lead_pt_mass_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_lead_pt_mass_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_lead_pt_mass_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_lead_pt_mass_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_lead_pt_mass_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("$p^{lead \ \gamma}_{T} / m_{\gamma\gamma}$")

# Save the plot
plt.savefig(f"lead_photon_to_mass_ratio_{suffix}.png")
print("~~~~~~~~~~~[INFO] lead photon pt to diphoton mass plot completed.~~~~~~~~~~~")
################################################################################################
########## sublead photon pt/mass (With data/MC ratio plot & data in errorbar style.) ##########
################################################################################################
print("~~~~~~~~~~~[INFO] sublead photon pt to diphoton mass plot started.~~~~~~~~~~~")
Data_sublead_pt_mass = ak.to_numpy(Data.sublead_pt / Data.mass)
Data_sublead_pt_mass_hist, bins = np.histogram(Data_sublead_pt_mass, range = (0.25, 0.65), bins = 70)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphoton.sublead_pt / Diphoton.mass, DYJets.sublead_pt / DYJets.mass, GJets.sublead_pt / GJets.mass, QCD.sublead_pt / QCD.mass, Top.sublead_pt / Top.mass, Diboson.sublead_pt / Diboson.mass]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_sublead_pt_mass = ak.to_numpy(WH_signal.sublead_pt / WH_signal.mass)
WH_signal_sublead_pt_mass_hist, _ = np.histogram(
    WH_signal_sublead_pt_mass,
    bins = bins,
    weights = WH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex = True, figsize = (10, 12))

# Upper plot: Data and MC comparison
# When you measure the number of events N, each event is independent and has a constant probability per unit time or per measurement. 
# Under these conditions, the probability distribution that describes the observed number of events is the Poisson distribution.
# Since the mean equals to variance in Poisson distribution, the statistical uncertainty is given by the square root of N.
axs[0].errorbar(
    x = bin_center, y = Data_sublead_pt_mass_hist,
    yerr = np.sqrt(Data_sublead_pt_mass_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_sublead_pt_mass, bins = bins, weights = WH_signal_weight * 100,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 100$"
)
axs[0].set_xlim((0.25, 0.65))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend()

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_sublead_pt_mass_hist, MC_hist,
    out = np.zeros_like(Data_sublead_pt_mass_hist, dtype=float),
    where = MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_sublead_pt_mass_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_sublead_pt_mass_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_sublead_pt_mass_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_sublead_pt_mass_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_sublead_pt_mass_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("$p^{sublead \ \gamma}_{T} / m_{\gamma\gamma}$")

# Save the plot
plt.savefig(f"sublead_photon_pt_to_mass_ratio_{suffix}.png")
print("~~~~~~~~~~~[INFO] sublead photon pt to diphoton mass plot completed.~~~~~~~~~~~")
######################################################################################
########## Max gamma ID (With data/MC ratio plot & data in errorbar style.) ##########
######################################################################################
print("~~~~~~~~~~~[INFO] max. photon BDT ID score plot started.~~~~~~~~~~~")
Data_max_gamma_ID = ak.to_numpy(Data_max_gamma_ID)
Data_max_gamma_ID_hist, bins = np.histogram(Data_max_gamma_ID, range = (-0.4, 1), bins = 40)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphoton_max_gamma_ID, DYJets_max_gamma_ID, GJets_max_gamma_ID, QCD_max_gamma_ID, Top_max_gamma_ID, Diboson_max_gamma_ID]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_max_gamma_ID = ak.to_numpy(WH_signal_max_gamma_ID)
WH_signal_max_gamma_ID_hist, _ = np.histogram(
    WH_signal_max_gamma_ID,
    bins = bins,
    weights = WH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex = True, figsize = (10, 12))

# Upper plot: Data and MC comparison
axs[0].errorbar(
    x = bin_center, y = Data_max_gamma_ID_hist,
    yerr = np.sqrt(Data_max_gamma_ID_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_max_gamma_ID, bins = bins, weights = WH_signal_weight * 100,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 100$"
)
axs[0].set_xlim((-0.4, 1))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(ncol = 2, loc = "upper left")

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_max_gamma_ID_hist, MC_hist,
    out = np.zeros_like(Data_max_gamma_ID_hist, dtype=float),
    where = MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_max_gamma_ID_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_max_gamma_ID_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_max_gamma_ID_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_max_gamma_ID_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_max_gamma_ID_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("$Max \ {\gamma} \ ID$")

# Save the plot
plt.savefig(f"max_gamma_ID_{suffix}.png")
print("~~~~~~~~~~~[INFO] max. photon BDT ID score plot completed.~~~~~~~~~~~")
######################################################################################
########## Min gamma ID (With data/MC ratio plot & data in errorbar style.) ##########
######################################################################################
print("~~~~~~~~~~~[INFO] min. photon BDT ID score plot started.~~~~~~~~~~~")
Data_min_gamma_ID = ak.to_numpy(Data_min_gamma_ID)
Data_min_gamma_ID_hist, bins = np.histogram(Data_min_gamma_ID, range = (-0.4, 1), bins = 40)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphoton_min_gamma_ID, DYJets_min_gamma_ID, GJets_min_gamma_ID, QCD_min_gamma_ID, Top_min_gamma_ID, Diboson_min_gamma_ID]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_min_gamma_ID = ak.to_numpy(WH_signal_min_gamma_ID)
WH_signal_min_gamma_ID_hist, _ = np.histogram(
    WH_signal_min_gamma_ID,
    bins = bins,
    weights = WH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize = (10, 12))

# Upper plot: Data and MC comparison
axs[0].errorbar(
    x = bin_center, y = Data_min_gamma_ID_hist,
    yerr = np.sqrt(Data_min_gamma_ID_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_min_gamma_ID, bins = bins, weights = WH_signal_weight * 100,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 100$"
)
axs[0].set_xlim((-0.4, 1))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(ncol = 2, loc = "upper left")

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_min_gamma_ID_hist, MC_hist,
    out = np.zeros_like(Data_min_gamma_ID_hist, dtype=float),
    where = MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_min_gamma_ID_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_min_gamma_ID_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_min_gamma_ID_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_min_gamma_ID_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_min_gamma_ID_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("$Min \ {\gamma} \ ID$")

# Save the plot
plt.savefig(f"min_gamma_ID_{suffix}.png")
print("~~~~~~~~~~~[INFO] min. photon BDT ID score plot completed.~~~~~~~~~~~")
#########################################################################################
########## lead photon eta (With data/MC ratio plot & data in errorbar style.) ##########
#########################################################################################
print("~~~~~~~~~~~[INFO] lead photon eta plot started.~~~~~~~~~~~")
Data_lead_gamma_eta = ak.to_numpy(Data.lead_eta)
Data_lead_gamma_eta_hist, bins = np.histogram(Data_lead_gamma_eta, range = (-3, 3), bins = 60)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphoton.lead_eta, DYJets.lead_eta, GJets.lead_eta, QCD.lead_eta, Top.lead_eta, Diboson.lead_eta]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_lead_gamma_eta = ak.to_numpy(WH_signal.lead_eta)
WH_signal_lead_gamma_eta_hist, _ = np.histogram(
    WH_signal_lead_gamma_eta,
    bins = bins,
    weights = WH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize = (10, 12))

# Upper plot: Data and MC comparison
axs[0].errorbar(
    x = bin_center, y = Data_lead_gamma_eta_hist,
    yerr = np.sqrt(Data_lead_gamma_eta_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_lead_gamma_eta, bins = bins, weights = WH_signal_weight * 100,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 100$"
)
axs[0].set_xlim((-3, 3))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(ncol = 2, loc = "upper right")

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_lead_gamma_eta_hist, MC_hist,
    out=np.zeros_like(Data_lead_gamma_eta_hist, dtype=float),
    where=MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_lead_gamma_eta_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_lead_gamma_eta_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_lead_gamma_eta_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_lead_gamma_eta_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_lead_gamma_eta_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("${\eta}^{lead \ \gamma}$")

# Save the plot
plt.savefig(f"lead_gamma_eta_{suffix}.png")
print("~~~~~~~~~~~[INFO] lead photon eta plot completed.~~~~~~~~~~~")
############################################################################################
########## sublead photon eta (With data/MC ratio plot & data in errorbar style.) ##########
############################################################################################
print("~~~~~~~~~~~[INFO] sublead photon eta plot started.~~~~~~~~~~~")
Data_sublead_gamma_eta = ak.to_numpy(Data.sublead_eta)
Data_sublead_gamma_eta_hist, bins = np.histogram(Data_sublead_gamma_eta, range = (-3, 3), bins = 60)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphoton.sublead_eta, DYJets.sublead_eta, GJets.sublead_eta, QCD.sublead_eta, Top.sublead_eta, Diboson.sublead_eta]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_sublead_gamma_eta = ak.to_numpy(WH_signal.sublead_eta)
WH_signal_sublead_gamma_eta_hist, _ = np.histogram(
    WH_signal_sublead_gamma_eta,
    bins = bins,
    weights = WH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize = (10, 12))

# Upper plot: Data and MC comparison
axs[0].errorbar(
    x = bin_center, y = Data_sublead_gamma_eta_hist,
    yerr = np.sqrt(Data_sublead_gamma_eta_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_sublead_gamma_eta, bins = bins, weights = WH_signal_weight * 100,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 100$"
)
axs[0].set_xlim((-3, 3))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(ncol = 2, loc = "upper right")

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_sublead_gamma_eta_hist, MC_hist,
    out=np.zeros_like(Data_sublead_gamma_eta_hist, dtype=float),
    where=MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_sublead_gamma_eta_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_sublead_gamma_eta_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_sublead_gamma_eta_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_sublead_gamma_eta_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_sublead_gamma_eta_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("${\eta}_{sublead \ \gamma}$")

# Save the plot
plt.savefig(f"sublead_gamma_eta_{suffix}.png")
print("~~~~~~~~~~~[INFO] sublead photon eta plot completed.~~~~~~~~~~~")
################################################################################
########## MET pt (With data/MC ratio plot & data in errorbar style.) ##########
################################################################################
print("~~~~~~~~~~~[INFO] MET pt plot started.~~~~~~~~~~~")
Data_met_pt = ak.to_numpy(Data.MET_pt)
Data_met_pt_hist, bins = np.histogram(Data_met_pt, range = (75, 180), bins = 50)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphoton.MET_pt, DYJets.MET_pt, GJets.MET_pt, QCD.MET_pt, Top.MET_pt, Diboson.MET_pt]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_MET_pt = ak.to_numpy(WH_signal.MET_pt)
WH_signal_MET_pt_hist, _ = np.histogram(
    WH_signal_MET_pt,
    bins = bins,
    weights = WH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize = (10, 12))

# Upper plot: Data and MC comparison
axs[0].errorbar(
    x = bin_center, y = Data_met_pt_hist,
    yerr = np.sqrt(Data_met_pt_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_MET_pt, bins = bins, weights = WH_signal_weight * 100,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 100$"
)
axs[0].set_xlim((75, 180))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(ncol = 2, loc = "upper right")

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_met_pt_hist, MC_hist,
    out=np.zeros_like(Data_met_pt_hist, dtype=float),
    where=MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_met_pt_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_met_pt_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_met_pt_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_met_pt_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_met_pt_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("$p^{miss}_{T} \ (GeV)$")

# Save the plot
plt.savefig(f"MET_pt_{suffix}.png")
print("~~~~~~~~~~~[INFO] MET pt plot completed.~~~~~~~~~~~")
########################################################################################
########## Number of jets (With data/MC ratio plot & data in errorbar style.) ##########
########################################################################################
print("~~~~~~~~~~~[INFO] jet multiplicity plot started.~~~~~~~~~~~")
Data_n_jets = ak.to_numpy(Data.n_jets)
Data_n_jets_hist, bins = np.histogram(Data_n_jets, range = (0, 8), bins = 8)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphoton.n_jets, DYJets.n_jets, GJets.n_jets, QCD.n_jets, Top.n_jets, Diboson.n_jets]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_n_jets = ak.to_numpy(WH_signal.n_jets)
WH_signal_n_jets_hist, _ = np.histogram(
    WH_signal_n_jets,
    bins = bins,
    weights = WH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize = (10, 12))

# Upper plot: Data and MC comparison
axs[0].errorbar(
    x = bin_center, y = Data_n_jets_hist,
    yerr = np.sqrt(Data_n_jets_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_n_jets, bins = bins, weights = WH_signal_weight * 100,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 100$"
)
axs[0].set_xlim((0, 8))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(ncol = 2, loc = "upper right")

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_n_jets_hist, MC_hist,
    out=np.zeros_like(Data_n_jets_hist, dtype=float),
    where=MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_n_jets_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_n_jets_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_n_jets_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_n_jets_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_n_jets_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("$jet \ multiplicity$")

# Save the plot
plt.savefig(f"jet_multiplicity_{suffix}.png")
print("~~~~~~~~~~~[INFO] jet multiplicity plot completed.~~~~~~~~~~~")
#######################################################################################
########## W lepton pt (With data/MC ratio plot & data in errorbar style.) ##########
#######################################################################################
print("~~~~~~~~~~~[INFO] W lepton pt plot started.~~~~~~~~~~~")
Data_W_lepton_pt = ak.to_numpy(np.where(Data.n_electrons == 1, Data.electron0_pt, Data.muon0_pt))
Data_W_lepton_pt_hist, bins = np.histogram(Data_W_lepton_pt, range = (10, 200), bins = 60)
bin_center = (bins[1:] + bins[:-1]) / 2

Diphoton_W_lepton_pt = np.where(Diphoton.n_electrons == 1, Diphoton.electron0_pt, Diphoton.muon0_pt)
DYJets_W_lepton_pt   = np.where(DYJets.n_electrons == 1, DYJets.electron0_pt, DYJets.muon0_pt)
GJets_W_lepton_pt    = np.where(GJets.n_electrons == 1, GJets.electron0_pt, GJets.muon0_pt)
QCD_W_lepton_pt      = np.where(QCD.n_electrons == 1, QCD.electron0_pt, QCD.muon0_pt)
Top_W_lepton_pt      = np.where(Top.n_electrons == 1, Top.electron0_pt, Top.muon0_pt)
Diboson_W_lepton_pt  = np.where(Diboson.n_electrons == 1, Diboson.electron0_pt, Diboson.muon0_pt)

MC_samples = [Diphoton_W_lepton_pt, DYJets_W_lepton_pt, GJets_W_lepton_pt, QCD_W_lepton_pt, Top_W_lepton_pt, Diboson_W_lepton_pt]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize = (10, 12))

WH_signal_W_lepton_pt = np.where(WH_signal.n_electrons == 1, WH_signal.electron0_pt, WH_signal.muon0_pt)
WH_signal_W_lepton_pt_hist, _ = np.histogram(
    WH_signal_W_lepton_pt,
    bins = bins,
    weights = WH_signal_weight
)

# Upper plot: Data and MC comparison
axs[0].errorbar(
    x = bin_center, y = Data_W_lepton_pt_hist,
    yerr = np.sqrt(Data_W_lepton_pt_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_W_lepton_pt, bins = bins, weights = WH_signal_weight * 100,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 100$"
)
axs[0].set_xlim((10, 200))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(ncol = 2, loc = "upper right")

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_W_lepton_pt_hist, MC_hist,
    out=np.zeros_like(Data_W_lepton_pt_hist, dtype=float),
    where=MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_W_lepton_pt_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_W_lepton_pt_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_W_lepton_pt_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_W_lepton_pt_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_W_lepton_pt_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("$p_{T}^{\\ell} \ (GeV)$")

# Save the plot
plt.savefig(f"W_lepton_pt_{suffix}.png")
print("~~~~~~~~~~~[INFO] W lepton pt plot completed.~~~~~~~~~~~")
#######################################################################################
########## W lepton eta (With data/MC ratio plot & data in errorbar style.) ##########
#######################################################################################
print("~~~~~~~~~~~[INFO] W lepton eta plot started.~~~~~~~~~~~")
Data_W_lepton_eta = ak.to_numpy(np.where(Data.n_electrons == 1, Data.electron0_eta, Data.muon0_eta))
Data_W_lepton_eta_hist, bins = np.histogram(Data_W_lepton_eta, range = (-3, 3), bins = 60)
bin_center = (bins[1:] + bins[:-1]) / 2

Diphoton_W_lepton_eta = np.where(Diphoton.n_electrons == 1, Diphoton.electron0_eta, Diphoton.muon0_eta)
DYJets_W_lepton_eta   = np.where(DYJets.n_electrons == 1, DYJets.electron0_eta, DYJets.muon0_eta)
GJets_W_lepton_eta    = np.where(GJets.n_electrons == 1, GJets.electron0_eta, GJets.muon0_eta)
QCD_W_lepton_eta      = np.where(QCD.n_electrons == 1, QCD.electron0_eta, QCD.muon0_eta)
Top_W_lepton_eta      = np.where(Top.n_electrons == 1, Top.electron0_eta, Top.muon0_eta)
Diboson_W_lepton_eta  = np.where(Diboson.n_electrons == 1, Diboson.electron0_eta, Diboson.muon0_eta)

MC_samples = [Diphoton_W_lepton_eta, DYJets_W_lepton_eta, GJets_W_lepton_eta, QCD_W_lepton_eta, Top_W_lepton_eta, Diboson_W_lepton_eta]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize = (10, 12))

WH_signal_W_lepton_eta = np.where(WH_signal.n_electrons == 1, WH_signal.electron0_eta, WH_signal.muon0_eta)
WH_signal_W_lepton_eta_hist, _ = np.histogram(
    WH_signal_W_lepton_eta,
    bins = bins,
    weights = WH_signal_weight
)

# Upper plot: Data and MC comparison
axs[0].errorbar(
    x = bin_center, y = Data_W_lepton_eta_hist,
    yerr = np.sqrt(Data_W_lepton_pt_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_W_lepton_eta, bins = bins, weights = WH_signal_weight * 100,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 100$"
)
axs[0].set_xlim((-3, 3))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(ncol = 2, loc = "upper right")

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_W_lepton_eta_hist, MC_hist,
    out=np.zeros_like(Data_W_lepton_eta_hist, dtype=float),
    where=MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_W_lepton_eta_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_W_lepton_eta_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_W_lepton_eta_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_W_lepton_eta_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_W_lepton_eta_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("${\eta}_{\\ell}$")

# Save the plot
plt.savefig(f"W_lepton_eta_{suffix}.png")
print("~~~~~~~~~~~[INFO] W lepton eta plot completed.~~~~~~~~~~~")
###########################################################################################
########## W transverse mass (With data/MC ratio plot & data in errorbar style.) ##########
###########################################################################################
print("~~~~~~~~~~~[INFO] transverse mass plot started.~~~~~~~~~~~")
def Get_mT(lepton_pt, lepton_phi, ntuple):
    mT = np.sqrt(2 * lepton_pt * ntuple.MET_pt * (1 - np.cos(lepton_phi - ntuple.MET_phi)))
    return mT

Data_mT = ak.to_numpy(
    np.where(
        Data.n_electrons == 1,
        Get_mT(Data.electron0_pt, Data.electron0_phi, Data),
        Get_mT(Data.muon0_pt, Data.muon0_phi, Data)
    )
)
Data_mT_hist, bins = np.histogram(Data_mT, range = (0, 150), bins = 70)
bin_center = (bins[1:] + bins[:-1]) / 2

Diphoton_mT = np.where(
    Diphoton.n_electrons == 1,
    Get_mT(Diphoton.electron0_pt, Diphoton.electron0_phi, Diphoton),
    Get_mT(Diphoton.muon0_pt, Diphoton.muon0_phi, Diphoton)
)
DYJets_mT = np.where(
    DYJets.n_electrons == 1,
    Get_mT(DYJets.electron0_pt, DYJets.electron0_phi, DYJets),
    Get_mT(DYJets.muon0_pt, DYJets.muon0_phi, DYJets)
)
GJets_mT = np.where(
    GJets.n_electrons == 1,
    Get_mT(GJets.electron0_pt, GJets.electron0_phi, GJets),
    Get_mT(GJets.muon0_pt, GJets.muon0_phi, GJets)
)
QCD_mT = np.where(
    QCD.n_electrons == 1,
    Get_mT(QCD.electron0_pt, QCD.electron0_phi, QCD),
    Get_mT(QCD.muon0_pt, QCD.muon0_phi, QCD)
)
Top_mT = np.where(
    Top.n_electrons == 1,
    Get_mT(Top.electron0_pt, Top.electron0_phi, Top),
    Get_mT(Top.muon0_pt, Top.muon0_phi, Top)
)
Diboson_mT = np.where(
    Diboson.n_electrons == 1,
    Get_mT(Diboson.electron0_pt, Diboson.electron0_phi, Diboson),
    Get_mT(Diboson.muon0_pt, Diboson.muon0_phi, Diboson)
)
MC_samples = [Diphoton_mT, DYJets_mT, GJets_mT, QCD_mT, Top_mT, Diboson_mT]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_mT = ak.to_numpy(
    np.where(
        WH_signal.n_electrons == 1,
        Get_mT(WH_signal.electron0_pt, WH_signal.electron0_phi, WH_signal),
        Get_mT(WH_signal.muon0_pt, WH_signal.muon0_phi, WH_signal)
    )
)
WH_signal_mT_hist, _ = np.histogram(
    WH_signal_mT,
    bins = bins,
    weights = WH_signal_weight
)
fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex = True, figsize = (10, 12))

# Upper plot: Data and MC comparison
# When you measure the number of events N, each event is independent and has a constant probability per unit time or per measurement. 
# Under these conditions, the probability distribution that describes the observed number of events is the Poisson distribution.
# Since the mean equals to variance in Poisson distribution, the statistical uncertainty is given by the square root of N.
axs[0].errorbar(
    x = bin_center, y = Data_mT_hist,
    yerr = np.sqrt(Data_mT_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_mT, bins = bins, weights = WH_signal_weight * 100,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 100$"
)
axs[0].set_xlim((0, 150))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(ncol = 2, loc = "upper right")

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_mT_hist, MC_hist,
    out = np.zeros_like(Data_mT_hist, dtype=float),
    where = MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_mT_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_mT_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_mT_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_mT_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_mT_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("$m_{T}^{\\ell \\nu} \ (GeV)$")

# Save the plot
plt.savefig(f"transverse_mass_{suffix}.png")
print("~~~~~~~~~~~[INFO] transverse mass plot completed.~~~~~~~~~~~")
#################################################################################################
########## Delta phi (diphoton, W) (With data/MC ratio plot & data in errorbar style.) ##########
#################################################################################################
print("~~~~~~~~~~~[INFO] Delta phi (diphoton, W) plot started.~~~~~~~~~~~")
def Get_W_phi(lepton_pt, lepton_phi, ntuple):
    W_px = lepton_pt * np.cos(lepton_phi) + ntuple.MET_pt * np.cos(ntuple.MET_phi)
    W_py = lepton_pt * np.sin(lepton_phi) + ntuple.MET_pt * np.sin(ntuple.MET_phi)
    W_phi = np.arctan2(W_py, W_px)
    return W_phi

Data_W_phi = ak.to_numpy(
    np.where(
        Data.n_electrons == 1,
        Get_W_phi(Data.electron0_pt, Data.electron0_phi, Data),
        Get_W_phi(Data.muon0_pt, Data.muon0_phi, Data)
    )
)
Data_W_phi_hist, bins = np.histogram(Data_W_phi, range = (0, 3), bins = 70)
bin_center = (bins[1:] + bins[:-1]) / 2

Diphoton_W_phi = np.where(
    Diphoton.n_electrons == 1,
    Get_W_phi(Diphoton.electron0_pt, Diphoton.electron0_phi, Diphoton),
    Get_W_phi(Diphoton.muon0_pt, Diphoton.muon0_phi, Diphoton)
)
DYJets_W_phi = np.where(
    DYJets.n_electrons == 1,
    Get_W_phi(DYJets.electron0_pt, DYJets.electron0_phi, DYJets),
    Get_W_phi(DYJets.muon0_pt, DYJets.muon0_phi, DYJets)
)
GJets_W_phi = np.where(
    GJets.n_electrons == 1,
    Get_W_phi(GJets.electron0_pt, GJets.electron0_phi, GJets),
    Get_W_phi(GJets.muon0_pt, GJets.muon0_phi, GJets)
)
QCD_W_phi = np.where(
    QCD.n_electrons == 1,
    Get_W_phi(QCD.electron0_pt, QCD.electron0_phi, QCD),
    Get_W_phi(QCD.muon0_pt, QCD.muon0_phi, QCD)
)
Top_W_phi = np.where(
    Top.n_electrons == 1,
    Get_W_phi(Top.electron0_pt, Top.electron0_phi, Top),
    Get_W_phi(Top.muon0_pt, Top.muon0_phi, Top)
)
Diboson_W_phi = np.where(
    Diboson.n_electrons == 1,
    Get_W_phi(Diboson.electron0_pt, Diboson.electron0_phi, Diboson),
    Get_W_phi(Diboson.muon0_pt, Diboson.muon0_phi, Diboson)
)
MC_samples = [Diphoton_W_phi, DYJets_W_phi, GJets_W_phi, QCD_W_phi, Top_W_phi, Diboson_W_phi]
MC_weights = [Diphoton_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_W_phi = ak.to_numpy(
    np.where(
        WH_signal.n_electrons == 1,
        Get_W_phi(WH_signal.electron0_pt, WH_signal.electron0_phi, WH_signal),
        Get_W_phi(WH_signal.muon0_pt, WH_signal.muon0_phi, WH_signal)
    )
)
WH_signal_W_phi_hist, _ = np.histogram(
    WH_signal_W_phi,
    bins = bins,
    weights = WH_signal_weight
)
fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex = True, figsize = (10, 12))

# Upper plot: Data and MC comparison
# When you measure the number of events N, each event is independent and has a constant probability per unit time or per measurement. 
# Under these conditions, the probability distribution that describes the observed number of events is the Poisson distribution.
# Since the mean equals to variance in Poisson distribution, the statistical uncertainty is given by the square root of N.
axs[0].errorbar(
    x = bin_center, y = Data_W_phi_hist,
    yerr = np.sqrt(Data_W_phi_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
)
axs[0].hist(
    WH_signal_mT, bins = bins, weights = WH_signal_weight * 500,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 500$"
)
axs[0].set_xlim((0, 3))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(ncol = 2, loc = "upper right")

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_W_phi_hist, MC_hist,
    out = np.zeros_like(Data_W_phi_hist, dtype=float),
    where = MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_W_phi_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_W_phi_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_W_phi_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_W_phi_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_W_phi_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("$\Delta \phi (\gamma\gamma, \\ell \\nu)$")

# Save the plot
plt.savefig(f"delta_phi_diphoton_W_{suffix}.png")
print("~~~~~~~~~~~[INFO] Delta phi (diphoton, W) plot completed.~~~~~~~~~~~")