import awkward as ak
import json
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os

preEE_luminosity = 7.9804 * 1000
postEE_luminosity = 26.6717 * 1000
with open('/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/cross_sections_for_Ntuples_v3.json', 'r') as f:
    cross_sections = json.load(f)
with open('/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022preEE/2022preEE_total_sum_genWeight_using_Ntuples_v3.json', 'r') as f:
    preEE_sum_genWeight = json.load(f)
with open('/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022postEE/2022postEE_total_sum_genWeight_using_Ntuples_v3.json', 'r') as f:
    postEE_sum_genWeight = json.load(f)

def get_weight(ntuple_name, ntuple):
    ntuple_name_without_suffix = ntuple_name

    if "_preEE" in ntuple_name:
        ntuple_name_without_suffix = ntuple_name.replace("_preEE", "")
        luminosity = preEE_luminosity
        total_sum_genWeight = preEE_sum_genWeight[ntuple_name]
        # total_sum_genWeight = preEE_sum_genWeight.get(ntuple_name, 1)
    elif "_postEE" in ntuple_name:
        ntuple_name_without_suffix = ntuple_name.replace("_postEE", "")
        luminosity = postEE_luminosity
        total_sum_genWeight = postEE_sum_genWeight[ntuple_name]
        # total_sum_genWeight = postEE_sum_genWeight.get(ntuple_name, 1)
    else:
        luminosity = 1
        total_sum_genWeight = 1

    if ntuple_name_without_suffix in cross_sections:
        if "M_125" in ntuple_name_without_suffix:
            xsec = cross_sections[ntuple_name_without_suffix] * 0.0025 # 0.0025 is the branching ratio of H -> gamma gamma.
        else:
            xsec = cross_sections[ntuple_name_without_suffix]
        return luminosity * xsec * (ntuple.genWeight / total_sum_genWeight)
    else:
        return None

################################################################################################################################
########## Load the Ntuples of 2022 preEE and postEE data ##########
################################################################################################################################
ntuples_dict = {}
basepath = {
    "preEE" : "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022preEE/Ntuples_v3",
    "postEE" : "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022postEE/Ntuples_v3"
}
excluded_Ntuples = [
    "ggH", "ggH_powheg", "ttH", "VBF", "VH_bkg",
    "WH_preEE_with_Event_weights", "WH_postEE_with_Event_weights", "WH_signal_filtered", "ZH_Zto2L_signal", "ZH_Zto2Nu_signal"
]
excluded_dir = ["merged", "root"]

for tag, path in basepath.items():
    for category in os.listdir(basepath[tag]):
        if category not in excluded_Ntuples:
            category_path = os.path.join(basepath[tag], category)
            print(f"[INFO] {category_path} is loaded.")
            for process in os.listdir(category_path):
                if process not in excluded_dir and os.path.isdir(os.path.join(category_path, process)) is True:
                    ntuple_path = os.path.join(category_path, process, "nominal")
                    ntuple_name = process.replace("-", "_")
                    ntuples_dict[ntuple_name] = ak.from_parquet(ntuple_path)
                    print(f"[INFO] {ntuple_name} ntuples are loaded.")
                else:
                    print(f"[INFO] {process} is excluded.")
                    pass
        else:
            print(f"[INFO] {category} is excluded.")
            pass

print("[INFO] 2022 preEE and postEE background Ntuples are loaded.")
print("\n")
print(f"[INFO] ntuples_dict: \n {ntuples_dict.keys()}")
print("\n")

################################################################################################################################
########## Apply WH-leptonic selections on the 2022 data and MC ntuples ##########
################################################################################################################################
data_ntuples_dict = {}
bkg_MC_ntuples_dict = {}
sig_MC_ntuples_dict = {}

for name, ntuple in ntuples_dict.items():
    if "WminusH_Hto2G_WtoLNu_M_125" in name or "WplusH_Hto2G_WtoLNu_M_125" in name:
        ntuple = ntuple[(ntuple.mass > 100) & (ntuple.mass < 180)] # Diphoton mass range.
        ntuple = ntuple[(ntuple.n_electrons + ntuple.n_muons) == 1] # One-lepton cut.
        ntuple = ntuple[(ntuple.lead_mvaID > -0.4) & (ntuple.sublead_mvaID > -0.4)] # Photon id cut.
        # ntuple = ntuple[ntuple.muon0_pt > 15] # Additional muon pt cut for WH-leptonic category.
        sig_MC_ntuples_dict[name] = ntuple
    else:
        ntuple = ntuple[(ntuple.mass > 100) & (ntuple.mass < 180)] # Diphoton mass range.
        ntuple = ntuple[(ntuple.mass < 115) | (ntuple.mass > 135)] # Blind the Higgs signal region except WH_signal.
        ntuple = ntuple[(ntuple.n_electrons + ntuple.n_muons) == 1] # One-lepton cut.
        ntuple = ntuple[(ntuple.lead_mvaID > -0.4) & (ntuple.sublead_mvaID > -0.4)] # Photon id cut.
        # ntuple = ntuple[ntuple.muon0_pt > 15] # Additional muon pt cut for WH-leptonic category.
        if "Data" in name:
            data_ntuples_dict[name] = ntuple
        else:
            bkg_MC_ntuples_dict[name] = ntuple

print("===========================================================================")
print("[INFO] data_ntuples_dict:\n", data_ntuples_dict.keys())
print("===========================================================================")
print("[INFO] signal_MC_ntuples_dict:\n", sig_MC_ntuples_dict.keys())
print("===========================================================================")
print("[INFO] background_MC_ntuples_dict:\n", bkg_MC_ntuples_dict.keys())
print("===========================================================================")
print("\n")

################################################################################################################################
########## Categorize each background sample based on processes involved ##########
################################################################################################################################
Diphotons_ntuples_dict = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if "GG_Box_3Jets" in name}
DYJets_ntuples_dict    = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if ("DYto2L_2Jets" in name or "DYGto2LG_1Jets" in name)}
GJets_ntuples_dict     = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if "GJet" in name}
QCD_ntuples_dict       = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if "QCD" in name}
Top_ntuples_dict       = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if ("TGJets" in name or "TTG_1Jets" in name or "TTGG_0Jets" in name or "TTto" in name)}
Diboson_ntuples_dict   = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if ("WW" in name or "WZ" in name or "ZZ" in name or "WGtoLNuG" in name or "ZGto" in name)}
WH_signal_ntuples_dict = {name : ntuple for name, ntuple in sig_MC_ntuples_dict.items()}
print("===========================================================================")
print("[INFO] Categorization of background samples completed.")
print("===========================================================================")
print("\n")

################################################################################################################################
########## Concatenate the MC weights of the same process ##########
################################################################################################################################
Diphotons_weight = np.concatenate([get_weight(name, nt) for name, nt in Diphotons_ntuples_dict.items()])
DYJets_weight    = np.concatenate([get_weight(name, nt) for name, nt in DYJets_ntuples_dict.items()])
GJets_weight     = np.concatenate([get_weight(name, nt) for name, nt in GJets_ntuples_dict.items()])
QCD_weight       = np.concatenate([get_weight(name, nt) for name, nt in QCD_ntuples_dict.items()])
Top_weight       = np.concatenate([get_weight(name, nt) for name, nt in Top_ntuples_dict.items()])
Diboson_weight   = np.concatenate([get_weight(name, nt) for name, nt in Diboson_ntuples_dict.items()])
WH_signal_weight = np.concatenate([get_weight(name, nt) for name, nt in WH_signal_ntuples_dict.items()])
print("===========================================================================")
print("[INFO] MC weights concatenation completed.")
print("===========================================================================")
print("\n")
################################################################################################################################
########## Max and Min gamma mvaID ##########
################################################################################################################################
Data_max_gamma_ID      = ak.to_numpy(np.concatenate([np.maximum(data_ntuples_dict[name].lead_mvaID, data_ntuples_dict[name].sublead_mvaID) for name in data_ntuples_dict.keys()]))
Data_min_gamma_ID      = ak.to_numpy(np.concatenate([np.minimum(data_ntuples_dict[name].lead_mvaID, data_ntuples_dict[name].sublead_mvaID) for name in data_ntuples_dict.keys()]))

Diphotons_max_gamma_ID = ak.to_numpy(np.concatenate([np.maximum(Diphotons_ntuples_dict[name].lead_mvaID, Diphotons_ntuples_dict[name].sublead_mvaID) for name in Diphotons_ntuples_dict.keys()]))
Diphotons_min_gamma_ID = ak.to_numpy(np.concatenate([np.minimum(Diphotons_ntuples_dict[name].lead_mvaID, Diphotons_ntuples_dict[name].sublead_mvaID) for name in Diphotons_ntuples_dict.keys()]))

DYJets_max_gamma_ID    = ak.to_numpy(np.concatenate([np.maximum(DYJets_ntuples_dict[name].lead_mvaID, DYJets_ntuples_dict[name].sublead_mvaID) for name in DYJets_ntuples_dict.keys()]))
DYJets_min_gamma_ID    = ak.to_numpy(np.concatenate([np.minimum(DYJets_ntuples_dict[name].lead_mvaID, DYJets_ntuples_dict[name].sublead_mvaID) for name in DYJets_ntuples_dict.keys()]))

GJets_max_gamma_ID     = ak.to_numpy(np.concatenate([np.maximum(GJets_ntuples_dict[name].lead_mvaID, GJets_ntuples_dict[name].sublead_mvaID) for name in GJets_ntuples_dict.keys()]))
GJets_min_gamma_ID     = ak.to_numpy(np.concatenate([np.minimum(GJets_ntuples_dict[name].lead_mvaID, GJets_ntuples_dict[name].sublead_mvaID) for name in GJets_ntuples_dict.keys()]))

QCD_max_gamma_ID       = ak.to_numpy(np.concatenate([np.maximum(QCD_ntuples_dict[name].lead_mvaID, QCD_ntuples_dict[name].sublead_mvaID) for name in QCD_ntuples_dict.keys()]))
QCD_min_gamma_ID       = ak.to_numpy(np.concatenate([np.minimum(QCD_ntuples_dict[name].lead_mvaID, QCD_ntuples_dict[name].sublead_mvaID) for name in QCD_ntuples_dict.keys()]))

Top_max_gamma_ID       = ak.to_numpy(np.concatenate([np.maximum(Top_ntuples_dict[name].lead_mvaID, Top_ntuples_dict[name].sublead_mvaID) for name in Top_ntuples_dict.keys()]))
Top_min_gamma_ID       = ak.to_numpy(np.concatenate([np.minimum(Top_ntuples_dict[name].lead_mvaID, Top_ntuples_dict[name].sublead_mvaID) for name in Top_ntuples_dict.keys()]))

Diboson_max_gamma_ID   = ak.to_numpy(np.concatenate([np.maximum(Diboson_ntuples_dict[name].lead_mvaID, Diboson_ntuples_dict[name].sublead_mvaID) for name in Diboson_ntuples_dict.keys()]))
Diboson_min_gamma_ID   = ak.to_numpy(np.concatenate([np.minimum(Diboson_ntuples_dict[name].lead_mvaID, Diboson_ntuples_dict[name].sublead_mvaID) for name in Diboson_ntuples_dict.keys()]))
################################################################################################################################
########## Labels for each MC sample ##########
################################################################################################################################
labels = ["Diphoton", "DYJets", r"$\it{\gamma}+jets$", "QCD", "Top", r"$VV/V+\it{\gamma}$"]
################################################################################################################################
########## Plotting ##########
################################################################################################################################
hep.style.use("CMS")
suffix = "20250621"
#######################################################################################
########## Diphoton mass (With data/MC ratio plot & data in errorbar style.) ##########
#######################################################################################
# Data_ggmass = ak.to_numpy(np.concatenate([data_ntuples_dict[name].mass for name in data_ntuples_dict.keys()]))
# Data_ggmass_hist, bins = np.histogram(Data_ggmass, range = (100, 180), bins = 70)
# bin_center = (bins[1:] + bins[:-1]) / 2

# Diphotons_ggmass = np.concatenate([Diphotons_ntuples_dict[name].mass for name in Diphotons_ntuples_dict.keys()])
# DYJets_ggmass    = np.concatenate([DYJets_ntuples_dict[name].mass for name in DYJets_ntuples_dict.keys()])
# GJets_ggmass     = np.concatenate([GJets_ntuples_dict[name].mass for name in GJets_ntuples_dict.keys()])
# QCD_ggmass       = np.concatenate([QCD_ntuples_dict[name].mass for name in QCD_ntuples_dict.keys()])
# Top_ggmass       = np.concatenate([Top_ntuples_dict[name].mass for name in Top_ntuples_dict.keys()])
# Diboson_ggmass   = np.concatenate([Diboson_ntuples_dict[name].mass for name in Diboson_ntuples_dict.keys()])

# MC_samples = [Diphotons_ggmass, DYJets_ggmass, GJets_ggmass, QCD_ggmass, Top_ggmass, Diboson_ggmass]
# MC_weights = [Diphotons_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]

# MC_hist, _ = np.histogram(
#     np.concatenate(MC_samples),
#     bins = bins,
#     weights = np.concatenate(MC_weights)
# )

# fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex = True, figsize = (9, 12))

# # Upper plot: Data and MC comparison
# # When you measure the number of events N, each event is independent and has a constant probability per unit time or per measurement. 
# # Under these conditions, the probability distribution that describes the observed number of events is the Poisson distribution.
# # Since the mean equals to variance in Poisson distribution, the statistical uncertainty is given by the square root of N.
# axs[0].errorbar(
#     x = bin_center, y = Data_ggmass_hist,
#     yerr = np.sqrt(Data_ggmass_hist), fmt = "ko", label = "Data"
# )
# axs[0].hist(
#     MC_samples, bins = bins, weights = MC_weights,
#     histtype = "stepfilled", stacked = True,
#     label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff", "#ffd700"]
# )

# axs[0].set_xlim((100, 180))
# axs[0].set_ylim(bottom = 0)
# axs[0].set_ylabel("Events", loc = "top")
# hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
# axs[0].legend(ncol = 2, loc = "upper right")

# # Lower plot: Data/MC ratio
# # Avoid division by zero
# ratio = np.divide(
#     Data_ggmass_hist, MC_hist,
#     out = np.zeros_like(Data_ggmass_hist, dtype=float),
#     where = MC_hist != 0
# )

# # Calculate errors for ratio
# data_errors = np.sqrt(Data_ggmass_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = np.zeros_like(ratio)
# nonzero_mask = (Data_ggmass_hist > 0) & (MC_hist > 0)
# ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
#     (data_errors[nonzero_mask] / Data_ggmass_hist[nonzero_mask])**2 +
#     (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
# )

# # data_errors = np.sqrt(Data_mass_hist)
# # mc_errors = np.sqrt(MC_hist)
# # ratio_errors = ratio * np.sqrt(
# #     (data_errors / Data_mass_hist)**2 + (mc_errors / MC_hist)**2
# # )

# axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
# axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
# axs[1].set_ylim(0, 2.0)
# axs[1].set_ylabel("Data/MC", loc = "center")
# axs[1].set_xlabel(r"$m_{\gamma\gamma}$")

# Save the plot
# plt.savefig(f"diphoton_mass_Ntuples_v3_{suffix}_without_one_lepton_and_mvaid_and_muon_pt_selections.png")
# plt.savefig(f"diphoton_mass_Ntuples_v3_{suffix}_without_muon_pt_selection.png")
# plt.savefig(f"diphoton_mass_Ntuples_v3_{suffix}_without_mvaid_selection.png")
# plt.savefig(f"diphoton_mass_Ntuples_v3_{suffix}.png")
###############################################################################################
########## Diphoton lead_pt/mass (With data/MC ratio plot & data in errorbar style.) ##########
###############################################################################################
Data_lead_pt_mass = ak.to_numpy(np.concatenate([(data_ntuples_dict[name].lead_pt / data_ntuples_dict[name].mass) for name in data_ntuples_dict.keys()]))
Data_lead_pt_mass_hist, bins = np.histogram(Data_lead_pt_mass, range = (0.3, 1.7), bins = 70)
bin_center = (bins[1:] + bins[:-1]) / 2

Diphotons_lead_pt_mass = np.concatenate([(Diphotons_ntuples_dict[name].lead_pt / Diphotons_ntuples_dict[name].mass) for name in Diphotons_ntuples_dict.keys()])
DYJets_lead_pt_mass    = np.concatenate([(DYJets_ntuples_dict[name].lead_pt / DYJets_ntuples_dict[name].mass) for name in DYJets_ntuples_dict.keys()])
GJets_lead_pt_mass     = np.concatenate([(GJets_ntuples_dict[name].lead_pt / GJets_ntuples_dict[name].mass) for name in GJets_ntuples_dict.keys()])
QCD_lead_pt_mass       = np.concatenate([(QCD_ntuples_dict[name].lead_pt / QCD_ntuples_dict[name].mass) for name in QCD_ntuples_dict.keys()])
Top_lead_pt_mass       = np.concatenate([(Top_ntuples_dict[name].lead_pt / Top_ntuples_dict[name].mass) for name in Top_ntuples_dict.keys()])
Diboson_lead_pt_mass   = np.concatenate([(Diboson_ntuples_dict[name].lead_pt / Diboson_ntuples_dict[name].mass) for name in Diboson_ntuples_dict.keys()])

MC_samples = [Diphotons_lead_pt_mass, DYJets_lead_pt_mass, GJets_lead_pt_mass, QCD_lead_pt_mass, Top_lead_pt_mass, Diboson_lead_pt_mass]
MC_weights = [Diphotons_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
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
plt.savefig(f"lead_photon_pt_to_mass_ratio_{suffix}.png")
##################################################################################################
########## Diphoton sublead_pt/mass (With data/MC ratio plot & data in errorbar style.) ##########
##################################################################################################
Data_sublead_pt_mass = ak.to_numpy(np.concatenate([(data_ntuples_dict[name].sublead_pt / data_ntuples_dict[name].mass) for name in data_ntuples_dict.keys()]))
Data_sublead_pt_mass_hist, bins = np.histogram(Data_sublead_pt_mass, range = (0.25, 0.65), bins = 70)
bin_center = (bins[1:] + bins[:-1]) / 2

Diphotons_sublead_pt_mass = np.concatenate([(Diphotons_ntuples_dict[name].sublead_pt / Diphotons_ntuples_dict[name].mass) for name in Diphotons_ntuples_dict.keys()])
DYJets_sublead_pt_mass    = np.concatenate([(DYJets_ntuples_dict[name].sublead_pt / DYJets_ntuples_dict[name].mass) for name in DYJets_ntuples_dict.keys()])
GJets_sublead_pt_mass     = np.concatenate([(GJets_ntuples_dict[name].sublead_pt / GJets_ntuples_dict[name].mass) for name in GJets_ntuples_dict.keys()])
QCD_sublead_pt_mass       = np.concatenate([(QCD_ntuples_dict[name].sublead_pt / QCD_ntuples_dict[name].mass) for name in QCD_ntuples_dict.keys()])
Top_sublead_pt_mass       = np.concatenate([(Top_ntuples_dict[name].sublead_pt / Top_ntuples_dict[name].mass) for name in Top_ntuples_dict.keys()])
Diboson_sublead_pt_mass   = np.concatenate([(Diboson_ntuples_dict[name].sublead_pt / Diboson_ntuples_dict[name].mass) for name in Diboson_ntuples_dict.keys()])

MC_samples = [Diphotons_sublead_pt_mass, DYJets_sublead_pt_mass, GJets_sublead_pt_mass, QCD_sublead_pt_mass, Top_sublead_pt_mass, Diboson_sublead_pt_mass]
MC_weights = [Diphotons_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
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
#########################################################################################
########## lead photon eta (With data/MC ratio plot & data in errorbar style.) ##########
#########################################################################################
Data_lead_gamma_eta = ak.to_numpy(np.concatenate([data_ntuples_dict[name].lead_eta for name in data_ntuples_dict.keys()]))
Data_lead_gamma_eta_hist, bins = np.histogram(Data_lead_gamma_eta, range = (-3, 3), bins = 60)
bin_center = (bins[1:] + bins[:-1]) / 2

Diphotons_lead_eta = np.concatenate([Diphotons_ntuples_dict[name].lead_eta for name in Diphotons_ntuples_dict.keys()])
DYJets_lead_eta    = np.concatenate([DYJets_ntuples_dict[name].lead_eta for name in DYJets_ntuples_dict.keys()])
GJets_lead_eta     = np.concatenate([GJets_ntuples_dict[name].lead_eta for name in GJets_ntuples_dict.keys()])
QCD_lead_eta       = np.concatenate([QCD_ntuples_dict[name].lead_eta for name in QCD_ntuples_dict.keys()])
Top_lead_eta       = np.concatenate([Top_ntuples_dict[name].lead_eta for name in Top_ntuples_dict.keys()])
Diboson_lead_eta   = np.concatenate([Diboson_ntuples_dict[name].lead_eta for name in Diboson_ntuples_dict.keys()])

MC_samples = [Diphotons_lead_eta, DYJets_lead_eta, GJets_lead_eta, QCD_lead_eta, Top_lead_eta, Diboson_lead_eta]
MC_weights = [Diphotons_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize=(10, 12))

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
axs[0].set_xlim((-3, 3))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend()

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
# # =======================================================================================================================================================
# # sublead photon eta (With data/MC ratio plot & data in errorbar style.)
# # =======================================================================================================================================================
Data_sublead_gamma_eta = ak.to_numpy(np.concatenate([data_ntuples_dict[name].sublead_eta for name in data_ntuples_dict.keys()]))
Data_sublead_gamma_eta_hist, bins = np.histogram(Data_sublead_gamma_eta, range = (-3, 3), bins = 60)
bin_center = (bins[1:] + bins[:-1]) / 2

Diphotons_sublead_eta = np.concatenate([Diphotons_ntuples_dict[name].sublead_eta for name in Diphotons_ntuples_dict.keys()])
DYJets_sublead_eta    = np.concatenate([DYJets_ntuples_dict[name].sublead_eta for name in DYJets_ntuples_dict.keys()])
GJets_sublead_eta     = np.concatenate([GJets_ntuples_dict[name].sublead_eta for name in GJets_ntuples_dict.keys()])
QCD_sublead_eta       = np.concatenate([QCD_ntuples_dict[name].sublead_eta for name in QCD_ntuples_dict.keys()])
Top_sublead_eta       = np.concatenate([Top_ntuples_dict[name].sublead_eta for name in Top_ntuples_dict.keys()])
Diboson_sublead_eta   = np.concatenate([Diboson_ntuples_dict[name].sublead_eta for name in Diboson_ntuples_dict.keys()])

MC_samples = [Diphotons_sublead_eta, DYJets_sublead_eta, GJets_sublead_eta, QCD_sublead_eta, Top_sublead_eta, Diboson_sublead_eta]
MC_weights = [Diphotons_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize=(10, 12))

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
axs[0].set_xlim((-3, 3))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend()

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
axs[1].set_xlabel("${\eta}^{sublead \ \gamma}$")

# Save the plot
plt.savefig(f"sublead_gamma_eta_{suffix}.png")
######################################################################################
########## Max gamma ID (With data/MC ratio plot & data in errorbar style.) ##########
######################################################################################
Data_max_gamma_ID_hist, bins = np.histogram(Data_max_gamma_ID, range = (-0.4, 1), bins = 40)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphotons_max_gamma_ID, DYJets_max_gamma_ID, GJets_max_gamma_ID, QCD_max_gamma_ID, Top_max_gamma_ID, Diboson_max_gamma_ID]
MC_weights = [Diphotons_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
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
axs[0].set_xlim((-0.4, 1))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend()

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
axs[1].set_xlabel("$Max {\ } {\gamma} {\ } ID$")

# Save the plot
plt.savefig(f"max_gamma_ID_{suffix}.png")
######################################################################################
########## Min gamma ID (With data/MC ratio plot & data in errorbar style.) ##########
######################################################################################
Data_min_gamma_ID_hist, bins = np.histogram(Data_min_gamma_ID, range = (-0.4, 1), bins = 40)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphotons_min_gamma_ID, DYJets_min_gamma_ID, GJets_min_gamma_ID, QCD_min_gamma_ID, Top_min_gamma_ID, Diboson_min_gamma_ID]
MC_weights = [Diphotons_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize=(10, 12))

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
axs[0].set_xlim((-0.4, 1))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend()

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
axs[1].set_xlabel("$Min {\ } {\gamma} {\ } ID$")

# Save the plot
plt.savefig(f"min_gamma_ID_{suffix}.png")
################################################################################
########## MET pt (With data/MC ratio plot & data in errorbar style.) ##########
################################################################################
Data_met_pt = ak.to_numpy(np.concatenate([data_ntuples_dict[name].MET_pt for name in data_ntuples_dict.keys()]))
Data_met_pt_hist, bins = np.histogram(Data_met_pt, range = (75, 180), bins = 50)
bin_center = (bins[1:] + bins[:-1]) / 2

Diphotons_met_pt = np.concatenate([Diphotons_ntuples_dict[name].MET_pt for name in Diphotons_ntuples_dict.keys()])
DYJets_met_pt    = np.concatenate([DYJets_ntuples_dict[name].MET_pt for name in DYJets_ntuples_dict.keys()])
GJets_met_pt     = np.concatenate([GJets_ntuples_dict[name].MET_pt for name in GJets_ntuples_dict.keys()])
QCD_met_pt       = np.concatenate([QCD_ntuples_dict[name].MET_pt for name in QCD_ntuples_dict.keys()])
Top_met_pt       = np.concatenate([Top_ntuples_dict[name].MET_pt for name in Top_ntuples_dict.keys()])
Diboson_met_pt   = np.concatenate([Diboson_ntuples_dict[name].MET_pt for name in Diboson_ntuples_dict.keys()])

WH_signal_met_pt = np.concatenate([WH_signal_ntuples_dict[name].MET_pt for name in WH_signal_ntuples_dict.keys()])

MC_samples = [Diphotons_met_pt, DYJets_met_pt, GJets_met_pt, QCD_met_pt, Top_met_pt, Diboson_met_pt]
MC_weights = [Diphotons_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]

MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_hist, _ = np.histogram(
    WH_signal_met_pt,
    bins = bins,
    weights = WH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize=(9, 12))

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
    WH_signal_hist, bins = bins, weights = WH_signal_weight * 50,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 50$"
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
axs[1].set_xlabel("$p_{T}^{miss}$")

# Save the plot
plt.savefig(f"MET_pt_Ntuples_v3_{suffix}.png")
########################################################################################
########## Number of jets (With data/MC ratio plot & data in errorbar style.) ##########
########################################################################################
Data_n_jets = ak.to_numpy(np.concatenate([data_ntuples_dict[name].n_jets for name in data_ntuples_dict.keys()]))
Data_n_jets_hist, bins = np.histogram(Data_n_jets, range = (0, 8), bins = 8)
bin_center = (bins[1:] + bins[:-1]) / 2

Diphotons_n_jets = np.concatenate([Diphotons_ntuples_dict[name].n_jets for name in Diphotons_ntuples_dict.keys()])
DYJets_n_jets    = np.concatenate([DYJets_ntuples_dict[name].n_jets for name in DYJets_ntuples_dict.keys()])
GJets_n_jets     = np.concatenate([GJets_ntuples_dict[name].n_jets for name in GJets_ntuples_dict.keys()])
QCD_n_jets       = np.concatenate([QCD_ntuples_dict[name].n_jets for name in QCD_ntuples_dict.keys()])
Top_n_jets       = np.concatenate([Top_ntuples_dict[name].n_jets for name in Top_ntuples_dict.keys()])
Diboson_n_jets   = np.concatenate([Diboson_ntuples_dict[name].n_jets for name in Diboson_ntuples_dict.keys()])

WH_signal_n_jets = np.concatenate([WH_signal_ntuples_dict[name].n_jets for name in WH_signal_ntuples_dict.keys()])

MC_samples = [Diphotons_n_jets, DYJets_n_jets, GJets_n_jets, QCD_n_jets, Top_n_jets, Diboson_n_jets]
MC_weights = [Diphotons_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight, Diboson_weight]

MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

WH_signal_hist, _ = np.histogram(
    WH_signal_n_jets,
    bins = bins,
    weights = WH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize=(9, 12))

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
    WH_signal_hist, bins = bins, weights = WH_signal_weight * 50,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 50$"
)
axs[0].set_xlim((0, 8))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend()

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
plt.savefig(f"jet_multiplicity_Ntuples_v3_{suffix}.png")