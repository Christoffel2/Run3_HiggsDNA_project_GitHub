import awkward as ak
import json
import matplotlib.pyplot as plt
import mplhep as hep
from math import sqrt, log
import numpy as np
import os
import pandas as pd
import vector
import xgboost as xgb

def get_dielectron_4momentum(sample):
    lead_4momentum = vector.array(
        {
            "pt" : sample.electron0_pt,
            "eta" : sample.electron0_eta,
            "phi" : sample.electron0_phi,
            "mass" : sample.electron0_mass
        }
    )
    sublead_4momentum = vector.array(
        {
            "pt" : sample.electron1_pt,
            "eta" : sample.electron1_eta,
            "phi" : sample.electron1_phi,
            "mass" : sample.electron1_mass
        }
    )
    return lead_4momentum + sublead_4momentum

def get_dimuon_4momentum(sample):
    lead_4momentum = vector.array(
        {
            "pt" : sample.muon0_pt,
            "eta" : sample.muon0_eta,
            "phi" : sample.muon0_phi,
            "mass" : sample.muon0_mass
        }
    )
    sublead_4momentum = vector.array(
        {
            "pt" : sample.muon1_pt,
            "eta" : sample.muon1_eta,
            "phi" : sample.muon1_phi,
            "mass" : sample.muon1_mass
        }
    )
    return lead_4momentum + sublead_4momentum

def get_electron_met_transverse_momentum(sample):
    lead_electron_transverse_momentum = vector.array(
        {
            "rho" : sample.electron0_pt,
            "phi" : sample.electron0_phi
        }
    )
    met_transverse_momentum = vector.array(
        {
            "rho" : sample.MET_pt,
            "phi" : sample.MET_phi
        }
    )
    return lead_electron_transverse_momentum + met_transverse_momentum

def get_muon_met_transverse_momentum(sample):
    lead_muon_transverse_momentum = vector.array(
        {
            "rho" : sample.muon0_pt,
            "phi" : sample.muon0_phi
        }
    )
    met_transverse_momentum = vector.array(
        {
            "rho" : sample.MET_pt,
            "phi" : sample.MET_phi
        }
    )
    return lead_muon_transverse_momentum + met_transverse_momentum

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
# Background and signal ntuples only. Excluding data ntuples.
signal_dict = {}
background_dict = {}

basepath = {
    "preEE"  : "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022preEE/Ntuples_v3",
    "postEE" : "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022postEE/Ntuples_v3"
}
print("===========================================================================")
for tag, path in basepath.items():
    for category in os.listdir(basepath[tag]):
        if category == "Data":
            print(f"2022-{tag} Data Ntuples are excluded.")
            pass
        elif category == "WH_signal":
            process_path = os.path.join(basepath[tag], category)
            if os.path.exists(process_path):
                for process in os.listdir(process_path):
                    ntuple_path = os.path.join(process_path, process, "nominal")
                    ntuple_name = process.replace("-", "_")
                    signal_dict[ntuple_name] = ak.from_parquet(ntuple_path)
                    print(f"2022-{tag} {ntuple_name}'s Ntuples are loaded (Signals).")
        else:
            process_path = os.path.join(basepath[tag], category)
            if os.path.exists(process_path):
                for process in os.listdir(process_path):
                    ntuple_path = os.path.join(process_path, process, "nominal")
                    ntuple_name = process.replace("-", "_")
                    background_dict[ntuple_name] = ak.from_parquet(ntuple_path)
                    print(f"2022-{tag} {ntuple_name}'s Ntuples are loaded (Backgrounds).")
print("===========================================================================")
print("\n")
print("===========================================================================")
print("Signal ntuples loaded.")
# print("Signal Ntuples:")
# print(signal_dict.keys())
# print(signal_dict.values())
print("===========================================================================")
print("\n")
print("===========================================================================")
print("Background ntuples loaded.")
# print("Background Ntuples:")
# print(background_dict.keys())
# print(background_dict.values())
print("===========================================================================")
print("\n")
################################################################################################################################
########## Apply WH-leptonic selections on the 2022 data and MC ntuples ##########
################################################################################################################################
WH_signal_dict = {}
WH_background_dict = {}

for name, ntuple in signal_dict.items():
    # Selection cuts on lepton id/iso, lepton pt, and deltaR between lepton and photon are included when making the ntuples.
    ntuple = ntuple[(ntuple.lead_mvaID > -0.4) & (ntuple.sublead_mvaID > -0.4)] # Photon id cut.
    ntuple = ntuple[(ntuple.n_electrons == 1) | (ntuple.n_muons == 1)] # One-lepton cut.
    WH_signal_dict[name] = ntuple

for name, ntuple in background_dict.items():
    # Selection cuts on lepton id/iso, lepton pt, and deltaR between lepton and photon are included when making the ntuples.
    ntuple = ntuple[(ntuple.lead_mvaID > -0.4) & (ntuple.sublead_mvaID > -0.4)] # Photon id cut.
    ntuple = ntuple[(ntuple.n_electrons == 1) | (ntuple.n_muons == 1)] # One-lepton cut.
    WH_background_dict[name] = ntuple

print("===========================================================================")
print(WH_signal_dict.keys())
print("===========================================================================")
print(WH_background_dict.keys())
print("===========================================================================")
print("\n")
################################################################################################################################
########## Load the XGBoost model ##########
################################################################################################################################
model_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/WH_leptonic_classifier_1500_3_0.05_10.json"
model = xgb.XGBClassifier()
model.load_model(model_path)

################################################################################################################################
########## Input variables for WH BDT ##########
################################################################################################################################
########## For WH-leptonic signals ##########
"""========== weight =========="""
WH_signal_weight = np.concatenate([get_weight(name, nt) for name, nt in WH_signal_dict.items()], axis = 0)
"""========== Photons =========="""
lead_photon_pt_mgg_ratio = np.concatenate([nt.lead_pt / nt.mass for nt in WH_signal_dict.values()], axis = 0)
sublead_photon_pt_mgg_ratio = np.concatenate([nt.sublead_pt / nt.mass for nt in WH_signal_dict.values()], axis = 0)
lead_photon_eta = np.concatenate([nt.lead_eta for nt in WH_signal_dict.values()], axis = 0)
sublead_photon_eta = np.concatenate([nt.sublead_eta for nt in WH_signal_dict.values()], axis = 0)
cos_delta_phi_photons = np.concatenate([np.cos(nt.lead_phi - nt.sublead_phi) for nt in WH_signal_dict.values()], axis = 0)
max_photon_id = np.concatenate([np.maximum(nt.lead_mvaID, nt.sublead_mvaID) for nt in WH_signal_dict.values()], axis = 0)
min_photon_id = np.concatenate([np.minimum(nt.lead_mvaID, nt.sublead_mvaID) for nt in WH_signal_dict.values()], axis = 0)
lead_photon_pixelSeed = np.concatenate([nt.lead_pixelSeed for nt in WH_signal_dict.values()], axis = 0)
sublead_photon_pixelSeed = np.concatenate([nt.sublead_pixelSeed for nt in WH_signal_dict.values()], axis = 0)
"""========== Leptons =========="""
lead_electron_pt = np.concatenate([nt.electron0_pt for nt in WH_signal_dict.values()], axis = 0)
lead_muon_pt = np.concatenate([nt.muon0_pt for nt in WH_signal_dict.values()], axis = 0)
lead_electron_eta = np.concatenate([nt.electron0_eta for nt in WH_signal_dict.values()], axis = 0)
lead_muon_eta = np.concatenate([nt.muon0_eta for nt in WH_signal_dict.values()], axis = 0)
delta_R_ld_photon_ld_electron = np.concatenate([np.sqrt((nt.lead_eta - nt.electron0_eta)**2 + (nt.lead_phi - nt.electron0_phi)**2) for nt in WH_signal_dict.values()], axis = 0)
delta_R_sld_photon_ld_electron = np.concatenate([np.sqrt((nt.sublead_eta - nt.electron0_eta)**2 + (nt.sublead_phi - nt.electron0_phi)**2) for nt in WH_signal_dict.values()], axis = 0)
delta_R_ld_photon_ld_muon = np.concatenate([np.sqrt((nt.lead_eta - nt.muon0_eta)**2 + (nt.lead_phi - nt.muon0_phi)**2) for nt in WH_signal_dict.values()], axis = 0)
delta_R_sld_photon_ld_muon = np.concatenate([np.sqrt((nt.sublead_eta - nt.muon0_eta)**2 + (nt.sublead_phi - nt.muon0_phi)**2) for nt in WH_signal_dict.values()], axis = 0)

# Lepton (here, electron + muon) information is obtained by concatenating the lead-electron and lead-muon arrays.
is_electron_event = np.concatenate([nt.n_electrons == 1 for nt in WH_signal_dict.values()])
is_muon_event = ~is_electron_event

lead_lepton_pt = np.where(is_electron_event, lead_electron_pt, lead_muon_pt)
lead_lepton_eta = np.where(is_electron_event, lead_electron_eta, lead_muon_eta)
delta_R_ld_photon_ld_lepton = np.where(is_electron_event, delta_R_ld_photon_ld_electron, delta_R_ld_photon_ld_muon)
delta_R_sld_photon_ld_lepton = np.where(is_electron_event, delta_R_sld_photon_ld_electron, delta_R_sld_photon_ld_muon)
"""========== Missing Transverse Energy =========="""
met_pt = np.concatenate([nt.MET_pt for nt in WH_signal_dict.values()], axis = 0)
met_sumEt = np.concatenate([nt.MET_sumEt for nt in WH_signal_dict.values()], axis = 0)
met_electron_Mt = np.concatenate([np.sqrt(2 * nt.electron0_pt * nt.MET_pt * (1 - np.cos(nt.electron0_phi - nt.MET_phi))) for nt in WH_signal_dict.values()], axis = 0)
met_muon_Mt = np.concatenate([np.sqrt(2 * nt.muon0_pt * nt.MET_pt * (1 - np.cos(nt.muon0_phi - nt.MET_phi))) for nt in WH_signal_dict.values()], axis = 0)

met_lepton_Mt = np.where(is_electron_event, met_electron_Mt, met_muon_Mt)
"""========== Jets =========="""
jet_multiplicity = np.concatenate([nt.n_jets for nt in WH_signal_dict.values()], axis = 0)
jet0_pt = np.concatenate([nt.Jet0_pt for nt in WH_signal_dict.values()], axis = 0)
jet1_pt = np.concatenate([nt.Jet1_pt for nt in WH_signal_dict.values()], axis = 0)
jet2_pt = np.concatenate([nt.Jet2_pt for nt in WH_signal_dict.values()], axis = 0)
jet3_pt = np.concatenate([nt.Jet3_pt for nt in WH_signal_dict.values()], axis = 0)
jet_max_btagPNetB = np.concatenate([np.maximum.reduce([nt.Jet0_btagPNetB, nt.Jet1_btagPNetB, nt.Jet2_btagPNetB, nt.Jet3_btagPNetB]) for nt in WH_signal_dict.values()], axis = 0)
jet_max_btagDeepFlavB = np.concatenate([np.maximum.reduce([nt.Jet0_btagDeepFlavB, nt.Jet1_btagDeepFlavB, nt.Jet2_btagDeepFlavB, nt.Jet3_btagDeepFlavB]) for nt in WH_signal_dict.values()], axis = 0)
jet_max_btagRobustParTAK4B = np.concatenate([np.maximum.reduce([nt.Jet0_btagRobustParTAK4B, nt.Jet1_btagRobustParTAK4B, nt.Jet2_btagRobustParTAK4B, nt.Jet3_btagRobustParTAK4B]) for nt in WH_signal_dict.values()], axis = 0)
"""========== WH Topology =========="""
delta_phi_diphoton_W_electron = np.concatenate([nt.phi - get_electron_met_transverse_momentum(nt).phi for nt in WH_signal_dict.values()], axis = 0)
delta_phi_diphoton_W_muon = np.concatenate([nt.phi - get_muon_met_transverse_momentum(nt).phi for nt in WH_signal_dict.values()], axis = 0)
WH_electron_pt_balance = np.concatenate([(nt.pt - get_electron_met_transverse_momentum(nt).rho) / nt.pt for nt in WH_signal_dict.values()], axis = 0)
WH_muon_pt_balance = np.concatenate([(nt.pt - get_muon_met_transverse_momentum(nt).rho) / nt.pt for nt in WH_signal_dict.values()], axis = 0)
Min_DPhi_W_Jets = np.concatenate([nt.Min_DPhi_W_Jets for nt in WH_signal_dict.values()], axis = 0)

delta_phi_diphoton_W_lepton = np.where(is_electron_event, delta_phi_diphoton_W_electron, delta_phi_diphoton_W_muon)
WH_lepton_pt_balance = np.where(is_electron_event, WH_electron_pt_balance, WH_muon_pt_balance)

WH_signal_combined = np.array(
    np.nan_to_num(
        [
            lead_photon_pt_mgg_ratio, sublead_photon_pt_mgg_ratio, lead_photon_eta, sublead_photon_eta, cos_delta_phi_photons, max_photon_id, min_photon_id, lead_photon_pixelSeed, sublead_photon_pixelSeed,
            lead_lepton_pt, lead_lepton_eta, delta_R_ld_photon_ld_lepton, delta_R_sld_photon_ld_lepton,
            met_pt, met_sumEt, met_lepton_Mt,
            jet_multiplicity, jet0_pt, jet1_pt, jet2_pt, jet3_pt, jet_max_btagPNetB, jet_max_btagDeepFlavB, jet_max_btagRobustParTAK4B,
            delta_phi_diphoton_W_lepton, Min_DPhi_W_Jets, WH_lepton_pt_balance
        ], nan = 0.0
    )
)
WH_signal_combined_transposed = WH_signal_combined.T

########## For WH-leptonic backgrounds ##########
WH_background_weight_list = []
WH_background_feature_list = []

for name, nt in WH_background_dict.items():
    weight = get_weight(name, nt)
    if weight is None or len(weight) != len(nt):
        print(f"[Warning] Skipping {name} due to invalid weight.")
        continue

    WH_background_weight_list.append(weight)

    # Append features in exactly the same order and condition as used to build WH_background_combined
    WH_background_feature_list.append((
        nt.lead_pt / nt.mass,
        nt.sublead_pt / nt.mass,
        nt.lead_eta,
        nt.sublead_eta,
        np.cos(nt.lead_phi - nt.sublead_phi),
        np.maximum(nt.lead_mvaID, nt.sublead_mvaID),
        np.minimum(nt.lead_mvaID, nt.sublead_mvaID),
        nt.lead_pixelSeed,
        nt.sublead_pixelSeed,
        np.where(nt.n_electrons == 1, nt.electron0_pt, nt.muon0_pt),
        np.where(nt.n_electrons == 1, nt.electron0_eta, nt.muon0_eta),
        np.where(nt.n_electrons == 1,
                 np.sqrt((nt.lead_eta - nt.electron0_eta)**2 + (nt.lead_phi - nt.electron0_phi)**2),
                 np.sqrt((nt.lead_eta - nt.muon0_eta)**2 + (nt.lead_phi - nt.muon0_phi)**2)),
        np.where(nt.n_electrons == 1,
                 np.sqrt((nt.sublead_eta - nt.electron0_eta)**2 + (nt.sublead_phi - nt.electron0_phi)**2),
                 np.sqrt((nt.sublead_eta - nt.muon0_eta)**2 + (nt.sublead_phi - nt.muon0_phi)**2)),
        nt.MET_pt,
        nt.MET_sumEt,
        np.where(nt.n_electrons == 1,
                 np.sqrt(2 * nt.electron0_pt * nt.MET_pt * (1 - np.cos(nt.electron0_phi - nt.MET_phi))),
                 np.sqrt(2 * nt.muon0_pt * nt.MET_pt * (1 - np.cos(nt.muon0_phi - nt.MET_phi)))),
        nt.n_jets,
        nt.Jet0_pt,
        nt.Jet1_pt,
        nt.Jet2_pt,
        nt.Jet3_pt,
        np.maximum.reduce([nt.Jet0_btagPNetB, nt.Jet1_btagPNetB, nt.Jet2_btagPNetB, nt.Jet3_btagPNetB]),
        np.maximum.reduce([nt.Jet0_btagDeepFlavB, nt.Jet1_btagDeepFlavB, nt.Jet2_btagDeepFlavB, nt.Jet3_btagDeepFlavB]),
        np.maximum.reduce([nt.Jet0_btagRobustParTAK4B, nt.Jet1_btagRobustParTAK4B, nt.Jet2_btagRobustParTAK4B, nt.Jet3_btagRobustParTAK4B]),
        np.where(nt.n_electrons == 1,
                 nt.phi - get_electron_met_transverse_momentum(nt).phi,
                 nt.phi - get_muon_met_transverse_momentum(nt).phi),
        nt.Min_DPhi_W_Jets,
        np.where(nt.n_electrons == 1,
                 (nt.pt - get_electron_met_transverse_momentum(nt).rho) / nt.pt,
                 (nt.pt - get_muon_met_transverse_momentum(nt).rho) / nt.pt),
    ))

# Final stacking
WH_background_weight = np.concatenate(WH_background_weight_list)
WH_background_combined = np.nan_to_num(np.array([np.concatenate([v[i] for v in WH_background_feature_list]) for i in range(27)]))
WH_background_combined_transposed = WH_background_combined.T

# """========== weight =========="""
# WH_background_weight = np.concatenate([get_weight(name, nt) for name, nt in WH_background_dict.items()], axis = 0)
# """========== Photons =========="""
# lead_photon_pt_mgg_ratio = np.concatenate([nt.lead_pt / nt.mass for nt in WH_background_dict.values()], axis = 0)
# sublead_photon_pt_mgg_ratio = np.concatenate([nt.sublead_pt / nt.mass for nt in WH_background_dict.values()], axis = 0)
# lead_photon_eta = np.concatenate([nt.lead_eta for nt in WH_background_dict.values()], axis = 0)
# sublead_photon_eta = np.concatenate([nt.sublead_eta for nt in WH_background_dict.values()], axis = 0)
# cos_delta_phi_photons = np.concatenate([np.cos(nt.lead_phi - nt.sublead_phi) for nt in WH_background_dict.values()], axis = 0)
# max_photon_id = np.concatenate([np.maximum(nt.lead_mvaID, nt.sublead_mvaID) for nt in WH_background_dict.values()], axis = 0)
# min_photon_id = np.concatenate([np.minimum(nt.lead_mvaID, nt.sublead_mvaID) for nt in WH_background_dict.values()], axis = 0)
# lead_photon_pixelSeed = np.concatenate([nt.lead_pixelSeed for nt in WH_background_dict.values()], axis = 0)
# sublead_photon_pixelSeed = np.concatenate([nt.sublead_pixelSeed for nt in WH_background_dict.values()], axis = 0)
# """========== Leptons =========="""
# lead_electron_pt = np.concatenate([nt.electron0_pt for nt in WH_background_dict.values()], axis = 0)
# lead_muon_pt = np.concatenate([nt.muon0_pt for nt in WH_background_dict.values()], axis = 0)
# lead_electron_eta = np.concatenate([nt.electron0_eta for nt in WH_background_dict.values()], axis = 0)
# lead_muon_eta = np.concatenate([nt.muon0_eta for nt in WH_background_dict.values()], axis = 0)
# delta_R_ld_photon_ld_electron = np.concatenate([np.sqrt((nt.lead_eta - nt.electron0_eta)**2 + (nt.lead_phi - nt.electron0_phi)**2) for nt in WH_background_dict.values()], axis = 0)
# delta_R_sld_photon_ld_electron = np.concatenate([np.sqrt((nt.sublead_eta - nt.electron0_eta)**2 + (nt.sublead_phi - nt.electron0_phi)**2) for nt in WH_background_dict.values()], axis = 0)
# delta_R_ld_photon_ld_muon = np.concatenate([np.sqrt((nt.lead_eta - nt.muon0_eta)**2 + (nt.lead_phi - nt.muon0_phi)**2) for nt in WH_background_dict.values()], axis = 0)
# delta_R_sld_photon_ld_muon = np.concatenate([np.sqrt((nt.sublead_eta - nt.muon0_eta)**2 + (nt.sublead_phi - nt.muon0_phi)**2) for nt in WH_background_dict.values()], axis = 0)

# # Lepton (here, electron + muon) information is obtained by concatenating the lead-electron and lead-muon arrays.
# is_electron_event = np.concatenate([nt.n_electrons == 1 for nt in WH_background_dict.values()])
# is_muon_event = ~is_electron_event

# lead_lepton_pt = np.where(is_electron_event, lead_electron_pt, lead_muon_pt)
# lead_lepton_eta = np.where(is_electron_event, lead_electron_eta, lead_muon_eta)
# delta_R_ld_photon_ld_lepton = np.where(is_electron_event, delta_R_ld_photon_ld_electron, delta_R_ld_photon_ld_muon)
# delta_R_sld_photon_ld_lepton = np.where(is_electron_event, delta_R_sld_photon_ld_electron, delta_R_sld_photon_ld_muon)
# """========== Missing Transverse Energy =========="""
# met_pt = np.concatenate([nt.MET_pt for nt in WH_background_dict.values()], axis = 0)
# met_sumEt = np.concatenate([nt.MET_sumEt for nt in WH_background_dict.values()], axis = 0)
# met_electron_Mt = np.concatenate([np.sqrt(2 * nt.electron0_pt * nt.MET_pt * (1 - np.cos(nt.electron0_phi - nt.MET_phi))) for nt in WH_background_dict.values()], axis = 0)
# met_muon_Mt = np.concatenate([np.sqrt(2 * nt.muon0_pt * nt.MET_pt * (1 - np.cos(nt.muon0_phi - nt.MET_phi))) for nt in WH_background_dict.values()], axis = 0)

# met_lepton_Mt = np.where(is_electron_event, met_electron_Mt, met_muon_Mt)
# """========== Jets =========="""
# jet_multiplicity = np.concatenate([nt.n_jets for nt in WH_background_dict.values()], axis = 0)
# jet0_pt = np.concatenate([nt.Jet0_pt for nt in WH_background_dict.values()], axis = 0)
# jet1_pt = np.concatenate([nt.Jet1_pt for nt in WH_background_dict.values()], axis = 0)
# jet2_pt = np.concatenate([nt.Jet2_pt for nt in WH_background_dict.values()], axis = 0)
# jet3_pt = np.concatenate([nt.Jet3_pt for nt in WH_background_dict.values()], axis = 0)
# jet_max_btagPNetB = np.concatenate([np.maximum.reduce([nt.Jet0_btagPNetB, nt.Jet1_btagPNetB, nt.Jet2_btagPNetB, nt.Jet3_btagPNetB]) for nt in WH_background_dict.values()], axis = 0)
# jet_max_btagDeepFlavB = np.concatenate([np.maximum.reduce([nt.Jet0_btagDeepFlavB, nt.Jet1_btagDeepFlavB, nt.Jet2_btagDeepFlavB, nt.Jet3_btagDeepFlavB]) for nt in WH_background_dict.values()], axis = 0)
# jet_max_btagRobustParTAK4B = np.concatenate([np.maximum.reduce([nt.Jet0_btagRobustParTAK4B, nt.Jet1_btagRobustParTAK4B, nt.Jet2_btagRobustParTAK4B, nt.Jet3_btagRobustParTAK4B]) for nt in WH_background_dict.values()], axis = 0)
# """========== WH Topology =========="""
# delta_phi_diphoton_W_electron = np.concatenate([nt.phi - get_electron_met_transverse_momentum(nt).phi for nt in WH_background_dict.values()], axis = 0)
# delta_phi_diphoton_W_muon = np.concatenate([nt.phi - get_muon_met_transverse_momentum(nt).phi for nt in WH_background_dict.values()], axis = 0)
# WH_electron_pt_balance = np.concatenate([(nt.pt - get_electron_met_transverse_momentum(nt).rho) / nt.pt for nt in WH_background_dict.values()], axis = 0)
# WH_muon_pt_balance = np.concatenate([(nt.pt - get_muon_met_transverse_momentum(nt).rho) / nt.pt for nt in WH_background_dict.values()], axis = 0)
# Min_DPhi_W_Jets = np.concatenate([nt.Min_DPhi_W_Jets for nt in WH_background_dict.values()], axis = 0)

# delta_phi_diphoton_W_lepton = np.where(is_electron_event, delta_phi_diphoton_W_electron, delta_phi_diphoton_W_muon)
# WH_lepton_pt_balance = np.where(is_electron_event, WH_electron_pt_balance, WH_muon_pt_balance)

# WH_background_combined = np.array(
#     np.nan_to_num(
#         [
#             lead_photon_pt_mgg_ratio, sublead_photon_pt_mgg_ratio, lead_photon_eta, sublead_photon_eta, cos_delta_phi_photons, max_photon_id, min_photon_id, lead_photon_pixelSeed, sublead_photon_pixelSeed,
#             lead_lepton_pt, lead_lepton_eta, delta_R_ld_photon_ld_lepton, delta_R_sld_photon_ld_lepton,
#             met_pt, met_sumEt, met_lepton_Mt,
#             jet_multiplicity, jet0_pt, jet1_pt, jet2_pt, jet3_pt, jet_max_btagPNetB, jet_max_btagDeepFlavB, jet_max_btagRobustParTAK4B,
#             delta_phi_diphoton_W_lepton, Min_DPhi_W_Jets, WH_lepton_pt_balance
#         ], nan = 0.0
#     )
# )
# WH_background_combined_transposed = WH_background_combined.T

var = [
    "ldPho_pt_to_M",                  # f0
    "sldPho_pt_to_M",                 # f1
    "ldPho_eta",
    "sldPho_eta",
    "cos_delta_phi_ldPho_sldPho",
    "max_Pho_id",                     # f5
    "min_Pho_id",
    "ldPho_pixel_seed",
    "sldPho_pixel_seed",
    "ld_Lep_pt",
    "ld_Lep_eta",                     # f10
    "delta_R_ldPho_ldLep",
    "delta_R_sldPho_ldLep",
    "MET_pt",
    "MET_SumEt",
    "MET_Lep_mt",                     # f15
    "Jet_multiplicity",
    "Jet0_pt",
    "Jet1_pt",
    "Jet2_pt",
    "Jet3_pt",                        # f20
    "jet_max_btagPNetB",
    "jet_max_btagDeepFlavB",
    "jet_max_btagRobustParTAK4B",
    "delta_phi_diPho_W_Lep",
    "min_delta_phi_W_Jets",           # f25
    "WH_Lep_pt_balance"
]
################################################################################################################################
########## Continue: Apply BDT and Compute Significance ##########
################################################################################################################################
"""========== Predict BDT Scores =========="""
bdt_sig = model.predict_proba(WH_signal_combined_transposed)[:, 1]
bdt_bkg = model.predict_proba(WH_background_combined_transposed)[:, 1]

"""========== Asimov Significance Definition =========="""
def asimov_significance(S, B):
    Br = 10 # Background regularization term (Ref: The Higgs boson machine learning challenge)
    if B <= 0: return 0
    return sqrt(2 * ((S + B + Br) * log(1 + S / (B + Br)) - S))

"""========== Sweep BDT cut thresholds =========="""
# Symbol for the significance is "Z".
thresholds = np.linspace(0, 1, 200)
Z_list = []
best_cut = 0
best_Z = 0

for t in thresholds:
    S = np.sum(WH_signal_weight[bdt_sig > t])
    assert WH_background_combined_transposed.shape[0] == len(WH_background_weight)
    B = np.sum(WH_background_weight[bdt_bkg > t])
    Z = asimov_significance(S, B)
    Z_list.append(Z)
    if Z > best_Z:
        best_Z = Z
        best_cut = t

"""========== Plotting Asimov Significance vs. BDT cut =========="""
hep.style.use("CMS")
plt.figure(figsize=(10, 10))
# plt.hist(bdt_sig, bins=100, weights=WH_signal_weight, histtype='step', label="Signal", color='blue')
# plt.hist(bdt_bkg, bins=100, weights=WH_background_weight, histtype='step', label="Background", color='orange')
plt.plot(thresholds, Z_list, label="Asimov Significance $Z$")
plt.axvline(best_cut, color='red', linestyle='--', label=f"Best Cut = {best_cut:.3f}")
plt.xlabel("BDT Score Cut")
plt.ylabel("Asimov Significance $Z$")
plt.title("Optimizing BDT Cut for WH Leptonic")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("asimov_significance_scan.png")

"""========== Print optimal results =========="""
print(f"Best BDT cut: {best_cut:.3f}")
print(f"Maximum Asimov significance: Z = {best_Z:.2f}")

# --- For signal ---
sorted_sig_indices = np.argsort(bdt_sig)
bdt_sig_sorted = bdt_sig[sorted_sig_indices]
weight_sig_sorted = WH_signal_weight[sorted_sig_indices]

cum_sig_yield = np.cumsum(weight_sig_sorted[::-1])[::-1]

# --- For background ---
sorted_bkg_indices = np.argsort(bdt_bkg)
bdt_bkg_sorted = bdt_bkg[sorted_bkg_indices]
weight_bkg_sorted = WH_background_weight[sorted_bkg_indices]

cum_bkg_yield = np.cumsum(weight_bkg_sorted[::-1])[::-1]

# --- Plot ---
# --- Define histogram bins ---
bins = np.linspace(0, 1, 50)  # 50 bins between 0 and 1

# --- Compute binned histograms (weighted) ---
hist_sig, bin_edges = np.histogram(bdt_sig, bins=bins, weights=WH_signal_weight)
hist_bkg, _ = np.histogram(bdt_bkg, bins=bins, weights=WH_background_weight)

# --- Compute cumulative yields from high BDT to low (reverse cumsum) ---
cum_sig = np.cumsum(hist_sig[::-1])[::-1]
cum_bkg = np.cumsum(hist_bkg[::-1])[::-1]

for i in range(len(bins) - 1):
    print(f"BDT > {bins[i]:.2f}: Cumulative Signal = {cum_sig[i]:.2f}, Cumulative Background = {cum_bkg[i]:.2f}")

# --- Bin centers for plotting ---
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# --- Plot binned cumulative histograms ---
plt.figure(figsize=(10, 10))
plt.bar(bin_centers, cum_sig, width=np.diff(bins), align='center',
        alpha=0.5, label="Signal", color='blue', edgecolor='black')
plt.bar(bin_centers, cum_bkg, width=np.diff(bins), align='center',
        alpha=0.5, label="Background", color='orange', edgecolor='black')

plt.xlabel("BDT Cut Threshold")
plt.ylabel("Cumulative Yield")
plt.yscale("log")
plt.title("Cumulative Yield Above BDT Cut")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cumulative_yield_binned_histogram.png")

# plt.figure(figsize=(10, 10))
# plt.step(bdt_sig_sorted, cum_sig_yield, where='post', label="Signal", color='blue')
# plt.step(bdt_bkg_sorted, cum_bkg_yield, where='post', label="Background", color='orange')
# plt.xlabel("BDT Cut Threshold")
# plt.ylabel("Cumulative Yield (Weight Sum > Cut)")
# plt.title("Cumulative Weighted Yield Above BDT Cut")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("cumulative_yield_vs_bdt_cut_hist.png")

# # --- Signal only ---
# plt.figure(figsize=(10, 10))
# plt.hist(bdt_sig, bins=100, weights=WH_signal_weight, histtype='step', color='blue', label='Signal')
# plt.xlabel("BDT Score")
# plt.ylabel("Weighted Yield")
# plt.title("Signal BDT Score Distribution")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("signal_bdt_distribution.png")

# # --- Background only ---
# plt.figure(figsize=(10, 10))
# plt.hist(bdt_bkg, bins=100, weights=WH_background_weight, histtype='step', color='orange', label='Background')
# plt.xlabel("BDT Score")
# plt.ylabel("Weighted Yield")
# plt.title("Background BDT Score Distribution")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("background_bdt_distribution.png")