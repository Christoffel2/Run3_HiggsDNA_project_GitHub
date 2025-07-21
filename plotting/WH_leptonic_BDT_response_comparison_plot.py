import awkward as ak
import json
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os
import vector
import xgboost

def electron_MET_transverse_momentum_vector(sample):
    lead_electron_transverse_momentum_vector = vector.array(
        {
            "rho" : sample.electron0_pt,
            "phi" : sample.electron0_phi
        }
    )
    MET_transverse_momentum_vector = vector.array(
        {
            "rho" : sample.MET_pt,
            "phi" : sample.MET_phi
        }
    )
    return lead_electron_transverse_momentum_vector + MET_transverse_momentum_vector

def muon_MET_transverse_momentum_vector(sample):
    lead_muon_transverse_momentum_vector = vector.array(
        {
            "rho" : sample.muon0_pt,
            "phi" : sample.muon0_phi
        }
    )
    MET_transverse_momentum_vector = vector.array(
        {
            "rho" : sample.MET_pt,
            "phi" : sample.MET_phi
        }
    )
    return lead_muon_transverse_momentum_vector + MET_transverse_momentum_vector

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
########## Load the Ntuples of 2022 preEE and postEE data and MC ##########
################################################################################################################################
ntuples_dict = {}

basepath = {
    "preEE"  : "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022preEE/Ntuples_v3",
    "postEE" : "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022postEE/Ntuples_v3"
}
excluded_Ntuples = [
    "ggH", "ggH_powheg", "ttH", "VBF", "VH_bkg", "VHtoGG_M-125_amcatnloFXFX_pythia8_preEE", "VHtoGG_M-125_amcatnloFXFX_pythia8_postEE",
    "WH_preEE_with_Event_weights", "WH_postEE_with_Event_weights", "WH_signal_filtered", "ZH_Zto2L_signal", "ZH_Zto2Nu_signal"
]
excluded_dir = ["merged", "root"]
print("===========================================================================")
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
print("===========================================================================")
print("\n")
print("===========================================================================")
print(ntuples_dict.keys()) 
# print(ntuples_dict.values())
print("===========================================================================")
print("\n")
################################################################################################################################
########## Apply WH-leptonic selections on the 2022 data and MC ntuples ##########
################################################################################################################################
data_ntuples_dict = {}
bkg_MC_ntuples_dict = {}
sig_MC_ntuples_dict = {}

for name, ntuple in ntuples_dict.items():
    # Selection cuts on lepton id/iso, lepton pt, and deltaR between lepton and photon are included when making the ntuples.
    if "WminusH_Hto2G_WtoLNu_M_125" in name or "WplusH_Hto2G_WtoLNu_M_125" in name:
        ntuple = ntuple[(ntuple.mass > 100) & (ntuple.mass < 180)]
        ntuple = ntuple[(ntuple.lead_mvaID > -0.4) & (ntuple.sublead_mvaID > -0.4)] # Photon id cut.
        ntuple = ntuple[(ntuple.n_electrons + ntuple.n_muons) == 1] # One-lepton cut.
        sig_MC_ntuples_dict[name] = ntuple
    else:
        ntuple = ntuple[(ntuple.mass > 100) & (ntuple.mass < 180)]
        ntuple = ntuple[(ntuple.mass < 115) | (ntuple.mass > 135)] # Blind the Higgs signal region except WH_signal.
        ntuple = ntuple[(ntuple.lead_mvaID > -0.4) & (ntuple.sublead_mvaID > -0.4)] # Photon id cut.
        ntuple = ntuple[(ntuple.n_electrons + ntuple.n_muons) == 1] # One-lepton cut.
        if "Data" in name:
            data_ntuples_dict[name] = ntuple
        else:
            bkg_MC_ntuples_dict[name] = ntuple
print("===========================================================================")
print(data_ntuples_dict.keys())
print("===========================================================================")
print(sig_MC_ntuples_dict.keys())
print("===========================================================================")
print(bkg_MC_ntuples_dict.keys())
print("===========================================================================")
print("\n")

Diphotons_ntuples_dict = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if "GG_Box_3Jets" in name}
DYJets_ntuples_dict = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if ("DYto2L_2Jets" in name or "DYGto2LG_1Jets" in name)}
GJets_ntuples_dict = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if "GJet" in name}
QCD_ntuples_dict = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if "QCD" in name}
Top_ntuples_dict = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if ("TGJets" in name or "TTG_1Jets" in name or "TTGG_0Jets" in name or "TTto" in name)}
Diboson_ntuples_dict = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if ("WW" in name or "WZ" in name or "ZZ" in name or "WGtoLNuG" in name or "ZGto" in name)}
WH_signal_ntuples_dict = {name : ntuple for name, ntuple in sig_MC_ntuples_dict.items()}

################################################################################################################################
########## Load the XGBoost model ##########
################################################################################################################################
model_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/WH_leptonic_classifier_1500_3_0.05_10_4.json"
model = xgboost.XGBClassifier()
model.load_model(model_path)
print("WH-leptonic BDT model is loaded.")
print("\n")

################################################################################################################################
########## Input variables for WH BDT ##########
################################################################################################################################
def feature_creation(sample_dict):
    if all("Data" in key for key in sample_dict.keys()):
        pass
    else:
        """========== weight =========="""
        weight = np.array(np.concatenate([get_weight(name, nt) for name, nt in sample_dict.items() if "Data" not in name], axis = 0))

    """========== Photons =========="""
    lead_photon_pt_mgg_ratio = np.concatenate([nt.lead_pt / nt.mass for nt in sample_dict.values()], axis = 0)
    sublead_photon_pt_mgg_ratio = np.concatenate([nt.sublead_pt / nt.mass for nt in sample_dict.values()], axis = 0)
    lead_photon_eta = np.concatenate([nt.lead_eta for nt in sample_dict.values()], axis = 0)
    sublead_photon_eta = np.concatenate([nt.sublead_eta for nt in sample_dict.values()], axis = 0)
    cos_delta_phi_photons = np.concatenate([np.cos(nt.lead_phi - nt.sublead_phi) for nt in sample_dict.values()], axis = 0)
    max_photon_id = np.concatenate([np.maximum(nt.lead_mvaID, nt.sublead_mvaID) for nt in sample_dict.values()], axis = 0)
    min_photon_id = np.concatenate([np.minimum(nt.lead_mvaID, nt.sublead_mvaID) for nt in sample_dict.values()], axis = 0)
    lead_photon_pixelSeed = np.concatenate([nt.lead_pixelSeed for nt in sample_dict.values()], axis = 0)
    sublead_photon_pixelSeed = np.concatenate([nt.sublead_pixelSeed for nt in sample_dict.values()], axis = 0)

    """========== Leptons =========="""
    lead_electron_pt = np.concatenate([nt.electron0_pt for nt in sample_dict.values()], axis = 0)
    lead_muon_pt = np.concatenate([nt.muon0_pt for nt in sample_dict.values()], axis = 0)
    lead_electron_eta = np.concatenate([nt.electron0_eta for nt in sample_dict.values()], axis = 0)
    lead_muon_eta = np.concatenate([nt.muon0_eta for nt in sample_dict.values()], axis = 0)
    delta_R_ld_photon_ld_electron = np.concatenate([np.sqrt((nt.lead_eta - nt.electron0_eta)**2 + (nt.lead_phi - nt.electron0_phi)**2) for nt in sample_dict.values()], axis = 0)
    delta_R_sld_photon_ld_electron = np.concatenate([np.sqrt((nt.sublead_eta - nt.electron0_eta)**2 + (nt.sublead_phi - nt.electron0_phi)**2) for nt in sample_dict.values()], axis = 0)
    delta_R_ld_photon_ld_muon = np.concatenate([np.sqrt((nt.lead_eta - nt.muon0_eta)**2 + (nt.lead_phi - nt.muon0_phi)**2) for nt in sample_dict.values()], axis = 0)
    delta_R_sld_photon_ld_muon = np.concatenate([np.sqrt((nt.sublead_eta - nt.muon0_eta)**2 + (nt.sublead_phi - nt.muon0_phi)**2) for nt in sample_dict.values()], axis = 0)

    # Lepton (here, electron + muon) information is obtained by concatenating the lead-electron and lead-muon arrays.
    is_electron_event = np.concatenate([nt.n_electrons == 1 for nt in sample_dict.values()])

    lead_lepton_pt = np.where(is_electron_event, lead_electron_pt, lead_muon_pt)
    lead_lepton_eta = np.where(is_electron_event, lead_electron_eta, lead_muon_eta)
    delta_R_ld_photon_ld_lepton = np.where(is_electron_event, delta_R_ld_photon_ld_electron, delta_R_ld_photon_ld_muon)
    delta_R_sld_photon_ld_lepton = np.where(is_electron_event, delta_R_sld_photon_ld_electron, delta_R_sld_photon_ld_muon)
    """========== Missing Transverse Energy =========="""
    met_pt = np.concatenate([nt.MET_pt for nt in sample_dict.values()], axis = 0)
    met_sumEt = np.concatenate([nt.MET_sumEt for nt in sample_dict.values()], axis = 0)
    met_electron_Mt = np.concatenate([np.sqrt(2 * nt.electron0_pt * nt.MET_pt * (1 - np.cos(nt.electron0_phi - nt.MET_phi))) for nt in sample_dict.values()], axis = 0)
    met_muon_Mt = np.concatenate([np.sqrt(2 * nt.muon0_pt * nt.MET_pt * (1 - np.cos(nt.muon0_phi - nt.MET_phi))) for nt in sample_dict.values()], axis = 0)

    met_lepton_Mt = np.where(is_electron_event, met_electron_Mt, met_muon_Mt)
    """========== Jets =========="""
    jet_multiplicity = np.concatenate([nt.n_jets for nt in sample_dict.values()], axis = 0)
    jet0_pt = np.concatenate([nt.Jet0_pt for nt in sample_dict.values()], axis = 0)
    # jet1_pt = np.concatenate([nt.Jet1_pt for nt in sample_dict.values()], axis = 0)
    # jet2_pt = np.concatenate([nt.Jet2_pt for nt in sample_dict.values()], axis = 0)
    # jet3_pt = np.concatenate([nt.Jet3_pt for nt in sample_dict.values()], axis = 0)
    jet_max_btagPNetB = np.concatenate([np.maximum.reduce([nt.Jet0_btagPNetB, nt.Jet1_btagPNetB, nt.Jet2_btagPNetB, nt.Jet3_btagPNetB]) for nt in sample_dict.values()], axis = 0)
    # jet_max_btagDeepFlavB = np.concatenate([np.maximum.reduce([nt.Jet0_btagDeepFlavB, nt.Jet1_btagDeepFlavB, nt.Jet2_btagDeepFlavB, nt.Jet3_btagDeepFlavB]) for nt in sample_dict.values()], axis = 0)
    # jet_max_btagRobustParTAK4B = np.concatenate([np.maximum.reduce([nt.Jet0_btagRobustParTAK4B, nt.Jet1_btagRobustParTAK4B, nt.Jet2_btagRobustParTAK4B, nt.Jet3_btagRobustParTAK4B]) for nt in sample_dict.values()], axis = 0)

    """========== WH Topology =========="""
    delta_phi_diphoton_W_electron = np.concatenate([nt.phi - electron_MET_transverse_momentum_vector(nt).phi for nt in sample_dict.values()], axis = 0)
    delta_phi_diphoton_W_muon = np.concatenate([nt.phi - muon_MET_transverse_momentum_vector(nt).phi for nt in sample_dict.values()], axis = 0)
    WH_electron_pt_balance = np.concatenate([(nt.pt - electron_MET_transverse_momentum_vector(nt).rho) / nt.pt for nt in sample_dict.values()], axis = 0)
    WH_muon_pt_balance = np.concatenate([(nt.pt - muon_MET_transverse_momentum_vector(nt).rho) / nt.pt for nt in sample_dict.values()], axis = 0)
    Min_DPhi_W_Jets = np.concatenate([nt.Min_DPhi_W_Jets for nt in sample_dict.values()], axis = 0)

    delta_phi_diphoton_W_lepton = np.where(is_electron_event, delta_phi_diphoton_W_electron, delta_phi_diphoton_W_muon)
    WH_lepton_pt_balance = np.where(is_electron_event, WH_electron_pt_balance, WH_muon_pt_balance)

    WH_combined = np.array(
        np.nan_to_num(
            [
                lead_photon_pt_mgg_ratio, sublead_photon_pt_mgg_ratio, lead_photon_eta, sublead_photon_eta, cos_delta_phi_photons, max_photon_id, min_photon_id, lead_photon_pixelSeed, sublead_photon_pixelSeed,
                lead_lepton_pt, lead_lepton_eta, delta_R_ld_photon_ld_lepton, delta_R_sld_photon_ld_lepton,
                met_pt, met_sumEt, met_lepton_Mt,
                jet_multiplicity, jet0_pt, 
                # jet1_pt, jet2_pt, jet3_pt, 
                jet_max_btagPNetB, 
                # jet_max_btagDeepFlavB, jet_max_btagRobustParTAK4B,
                delta_phi_diphoton_W_lepton, Min_DPhi_W_Jets, WH_lepton_pt_balance
            ], nan = 0.0
        )
    )
    WH_combined_transposed = WH_combined.T
    return WH_combined_transposed

data_features = feature_creation(data_ntuples_dict)
Diphotons_features = feature_creation(Diphotons_ntuples_dict)
DYJets_features = feature_creation(DYJets_ntuples_dict)
GJets_features = feature_creation(GJets_ntuples_dict)
# QCD_features = feature_creation(QCD_ntuples_dict) # No QCD after one-lepton selection.
Top_features = feature_creation(Top_ntuples_dict)
Diboson_features = feature_creation(Diboson_ntuples_dict)
WH_signal_features = feature_creation(WH_signal_ntuples_dict)

data_bdt_response = model.predict_proba(data_features)[: , 1]
Diphotons_bdt_response = model.predict_proba(Diphotons_features)[: , 1]
DYJets_bdt_response = model.predict_proba(DYJets_features)[: , 1]
GJets_bdt_response = model.predict_proba(GJets_features)[: , 1]
# QCD_bdt_response = model.predict_proba(QCD_features)[: , 1]
Top_bdt_response = model.predict_proba(Top_features)[: , 1]
Diboson_bdt_response = model.predict_proba(Diboson_features)[: , 1]
WH_signal_bdt_response = model.predict_proba(WH_signal_features)[: , 1]
print("Data, signal and bkg BDT responses are calculated.")

################################################################################################################################
########## Labels for each MC sample ##########
################################################################################################################################
labels = ["Diphoton", "DYJets", "$\it{\gamma}+jets$", "Top", "$VV/V+\it{\gamma}$"]
################################################################################################################################
########## Plotting ##########
################################################################################################################################
hep.style.use("CMS")
suffix = "Ntuples_v3_20250630"

# Data = np.concatenate([data_ntuples_dict.values()])
# Diphotons = np.concatenate([Diphotons_ntuples_dict.values()])
# DYJets = np.concatenate([DYJets_ntuples_dict.values()])
# GJets = np.concatenate([GJets_ntuples_dict.values()])
# QCD = np.concatenate([QCD_ntuples_dict.values()])
# Top = np.concatenate([Top_ntuples_dict.values()])
# Diboson = np.concatenate([Diboson_ntuples_dict.values()])

Diphotons_weight = np.concatenate([get_weight(name, nt) for name, nt in Diphotons_ntuples_dict.items()])
DYJets_weight = np.concatenate([get_weight(name, nt) for name, nt in DYJets_ntuples_dict.items()])
GJets_weight = np.concatenate([get_weight(name, nt) for name, nt in GJets_ntuples_dict.items()])
# QCD_weight = np.concatenate([get_weight(name, nt) for name, nt in QCD_ntuples_dict.items()])
Top_weight = np.concatenate([get_weight(name, nt) for name, nt in Top_ntuples_dict.items()])
Diboson_weight = np.concatenate([get_weight(name, nt) for name, nt in Diboson_ntuples_dict.items()])
WH_signal_weight = np.concatenate([get_weight(name, nt) for name, nt in WH_signal_ntuples_dict.items()])


Data_BDT_response_hist, bins = np.histogram(data_bdt_response, range = (0, 1), bins = 30)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphotons_bdt_response, DYJets_bdt_response, GJets_bdt_response, Top_bdt_response, Diboson_bdt_response]
MC_weights = [Diphotons_weight, DYJets_weight, GJets_weight, Top_weight, Diboson_weight]
MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)
# print(np.any(MC_hist == 0)) # Check if any bin is empty.
WH_signal_BDT_response_hist, _ = np.histogram(
    WH_signal_bdt_response,
    bins = bins,
    weights = WH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex = True, figsize = (10, 12))

# Upper plot: Data and MC comparison
# When you measure the number of events N, each event is independent and has a constant probability per unit time or per measurement. 
# Under these conditions, the probability distribution that describes the observed number of events is the Poisson distribution.
# Since the mean equals to variance in Poisson distribution, the statistical uncertainty is given by the square root of N.
axs[0].errorbar(
    x = bin_center, y = Data_BDT_response_hist,
    yerr = np.sqrt(Data_BDT_response_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#7b68ee", "#7cfc00", "#00bfff", "#ffd700"]
)
# normalized_WH_signal_bdt_response = WH_signal_bdt_response / np.sum(WH_signal_bdt_response) * np.sum(MC_hist)
axs[0].hist(
    WH_signal_bdt_response, bins = bins, weights = WH_signal_weight * 500,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$W(\\to \\ell\\nu)H(\\to \\gamma\\gamma) \\times 500$"
)
axs[0].set_xlim((0, 1))
axs[0].set_ylim(bottom = 1e-1, top = 1e3) # Set lower limit to 10^-1 to avoid log(0) issue.
axs[0].set_ylabel("Events", loc = "top")
axs[0].set_yscale("log")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 34.6521, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(ncol = 2, loc = "upper right", fontsize = 20, columnspacing = 1.5)

# Lower plot: Data/MC ratio
# Avoid division by zero
ratio = np.divide(
    Data_BDT_response_hist, MC_hist,
    out = np.zeros_like(Data_BDT_response_hist, dtype=float),
    where = MC_hist != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(Data_BDT_response_hist)
mc_errors = np.sqrt(MC_hist)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_BDT_response_hist > 0) & (MC_hist > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_BDT_response_hist[nonzero_mask])**2 +
    (mc_errors[nonzero_mask] / MC_hist[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_BDT_response_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_BDT_response_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("WH leptonic BDT score")

# Save the plot
plt.savefig(f"WH_leptonic_BDT_score_{suffix}.png")