import awkward as ak
import json
import numpy as np
import os
import vector
import argparse

parser = argparse.ArgumentParser(description = "Variables for WH BDT training.")

parser.add_argument(
    "-s",
    "--sample_name",
    type = str,
    help = "Specify the process name you want to load.",
    choices = ("Diphoton", "DYJets", "GJets", "QCD", "Top", "VV", "WG", "ZG", "VBF", "ggH", "ggH_powheg", "ttH", "WH_signal", "ZH_Zto2L_signal", "ZH_Zto2Nu_signal", "VH_bkg"),
)
parser.add_argument(
    "-o",
    "--output_filename",
    type = str,
    help = "Specify the name of the npy file.",
)

args = parser.parse_args()

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
sample_name = args.sample_name
ntuples_dict = {} # Since the available options of sample_name do not include Data, we do not need to add an additional if-else statement to check if the sample_name is Data or MC.

basepath = {
    "preEE"  : "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022preEE/Ntuples_v3",
    "postEE" : "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/2022postEE/Ntuples_v3"
}
print("===========================================================================")
for tag, path in basepath.items():
    process_path = os.path.join(basepath[tag], sample_name)
    if os.path.exists(process_path):
        for process in os.listdir(process_path):
            ntuple_path = os.path.join(process_path, process, "nominal")
            ntuple_name = process.replace("-", "_")
            ntuples_dict[ntuple_name] = ak.from_parquet(ntuple_path)
            print(f"2022-{tag} {ntuple_name}'s Ntuples are loaded.")
print("===========================================================================")
print("\n")
print("===========================================================================")
print(ntuples_dict.keys()) # Output should include what you enter in arg.sample_name.
print(ntuples_dict.values())
print("===========================================================================")
print("\n")
################################################################################################################################
########## Apply WH-leptonic selections on the 2022 data and MC ntuples ##########
################################################################################################################################
WH_ntuples_dict = {}

for name, ntuple in ntuples_dict.items():
    # Selection cuts on lepton id/iso, lepton pt, and deltaR between lepton and photon are included when making the ntuples.
    ntuple = ntuple[(ntuple.lead_mvaID > -0.4) & (ntuple.sublead_mvaID > -0.4)] # Photon id cut.
    ntuple = ntuple[(ntuple.n_electrons + ntuple.n_muons) == 1] # One-lepton cut.
    WH_ntuples_dict[name] = ntuple

print("===========================================================================")
print(f"WH-leptonic selections are applied on {WH_ntuples_dict.keys()}.")
print(WH_ntuples_dict.keys())
print("===========================================================================")
print("\n")
################################################################################################################################
########## Input variables for WH BDT ##########
################################################################################################################################
"""========== weight =========="""
weight = np.array(np.concatenate([get_weight(name, nt) for name, nt in WH_ntuples_dict.items()], axis = 0))

"""========== Photons =========="""
lead_photon_pt_mgg_ratio = np.concatenate([nt.lead_pt / nt.mass for nt in WH_ntuples_dict.values()], axis = 0)
sublead_photon_pt_mgg_ratio = np.concatenate([nt.sublead_pt / nt.mass for nt in WH_ntuples_dict.values()], axis = 0)
lead_photon_eta = np.concatenate([nt.lead_eta for nt in WH_ntuples_dict.values()], axis = 0)
sublead_photon_eta = np.concatenate([nt.sublead_eta for nt in WH_ntuples_dict.values()], axis = 0)
cos_delta_phi_photons = np.concatenate([np.cos(nt.lead_phi - nt.sublead_phi) for nt in WH_ntuples_dict.values()], axis = 0)
max_photon_id = np.concatenate([np.maximum(nt.lead_mvaID, nt.sublead_mvaID) for nt in WH_ntuples_dict.values()], axis = 0)
min_photon_id = np.concatenate([np.minimum(nt.lead_mvaID, nt.sublead_mvaID) for nt in WH_ntuples_dict.values()], axis = 0)
lead_photon_pixelSeed = np.concatenate([nt.lead_pixelSeed for nt in WH_ntuples_dict.values()], axis = 0)
sublead_photon_pixelSeed = np.concatenate([nt.sublead_pixelSeed for nt in WH_ntuples_dict.values()], axis = 0)

"""========== Leptons =========="""
lead_electron_pt = np.concatenate([nt.electron0_pt for nt in WH_ntuples_dict.values()], axis = 0)
lead_muon_pt = np.concatenate([nt.muon0_pt for nt in WH_ntuples_dict.values()], axis = 0)
lead_electron_eta = np.concatenate([nt.electron0_eta for nt in WH_ntuples_dict.values()], axis = 0)
lead_muon_eta = np.concatenate([nt.muon0_eta for nt in WH_ntuples_dict.values()], axis = 0)
delta_R_ld_photon_ld_electron = np.concatenate([np.sqrt((nt.lead_eta - nt.electron0_eta)**2 + (nt.lead_phi - nt.electron0_phi)**2) for nt in WH_ntuples_dict.values()], axis = 0)
delta_R_sld_photon_ld_electron = np.concatenate([np.sqrt((nt.sublead_eta - nt.electron0_eta)**2 + (nt.sublead_phi - nt.electron0_phi)**2) for nt in WH_ntuples_dict.values()], axis = 0)
delta_R_ld_photon_ld_muon = np.concatenate([np.sqrt((nt.lead_eta - nt.muon0_eta)**2 + (nt.lead_phi - nt.muon0_phi)**2) for nt in WH_ntuples_dict.values()], axis = 0)
delta_R_sld_photon_ld_muon = np.concatenate([np.sqrt((nt.sublead_eta - nt.muon0_eta)**2 + (nt.sublead_phi - nt.muon0_phi)**2) for nt in WH_ntuples_dict.values()], axis = 0)

# Lepton (here, electron + muon) information is obtained by concatenating the lead-electron and lead-muon arrays.
is_electron_event = np.concatenate([nt.n_electrons == 1 for nt in WH_ntuples_dict.values()])
is_muon_event = ~is_electron_event

lead_lepton_pt = np.where(is_electron_event, lead_electron_pt, lead_muon_pt)
lead_lepton_eta = np.where(is_electron_event, lead_electron_eta, lead_muon_eta)
delta_R_ld_photon_ld_lepton = np.where(is_electron_event, delta_R_ld_photon_ld_electron, delta_R_ld_photon_ld_muon)
delta_R_sld_photon_ld_lepton = np.where(is_electron_event, delta_R_sld_photon_ld_electron, delta_R_sld_photon_ld_muon)
"""========== Missing Transverse Energy =========="""
met_pt = np.concatenate([nt.MET_pt for nt in WH_ntuples_dict.values()], axis = 0)
met_sumEt = np.concatenate([nt.MET_sumEt for nt in WH_ntuples_dict.values()], axis = 0)
met_electron_Mt = np.concatenate([np.sqrt(2 * nt.electron0_pt * nt.MET_pt * (1 - np.cos(nt.electron0_phi - nt.MET_phi))) for nt in WH_ntuples_dict.values()], axis = 0)
met_muon_Mt = np.concatenate([np.sqrt(2 * nt.muon0_pt * nt.MET_pt * (1 - np.cos(nt.muon0_phi - nt.MET_phi))) for nt in WH_ntuples_dict.values()], axis = 0)

met_lepton_Mt = np.where(is_electron_event, met_electron_Mt, met_muon_Mt)
"""========== Jets =========="""
jet_multiplicity = np.concatenate([nt.n_jets for nt in WH_ntuples_dict.values()], axis = 0)
jet0_pt = np.concatenate([nt.Jet0_pt for nt in WH_ntuples_dict.values()], axis = 0)
# jet1_pt = np.concatenate([nt.Jet1_pt for nt in WH_ntuples_dict.values()], axis = 0)
# jet2_pt = np.concatenate([nt.Jet2_pt for nt in WH_ntuples_dict.values()], axis = 0)
# jet3_pt = np.concatenate([nt.Jet3_pt for nt in WH_ntuples_dict.values()], axis = 0)
jet_max_btagPNetB = np.concatenate([np.maximum.reduce([nt.Jet0_btagPNetB, nt.Jet1_btagPNetB, nt.Jet2_btagPNetB, nt.Jet3_btagPNetB]) for nt in WH_ntuples_dict.values()], axis = 0)
# jet_max_btagDeepFlavB = np.concatenate([np.maximum.reduce([nt.Jet0_btagDeepFlavB, nt.Jet1_btagDeepFlavB, nt.Jet2_btagDeepFlavB, nt.Jet3_btagDeepFlavB]) for nt in WH_ntuples_dict.values()], axis = 0)
# jet_max_btagRobustParTAK4B = np.concatenate([np.maximum.reduce([nt.Jet0_btagRobustParTAK4B, nt.Jet1_btagRobustParTAK4B, nt.Jet2_btagRobustParTAK4B, nt.Jet3_btagRobustParTAK4B]) for nt in WH_ntuples_dict.values()], axis = 0)

"""========== WH Topology =========="""
delta_phi_diphoton_W_electron = np.concatenate([nt.phi - get_electron_met_transverse_momentum(nt).phi for nt in WH_ntuples_dict.values()], axis = 0)
delta_phi_diphoton_W_muon = np.concatenate([nt.phi - get_muon_met_transverse_momentum(nt).phi for nt in WH_ntuples_dict.values()], axis = 0)
WH_electron_pt_balance = np.concatenate([(nt.pt - get_electron_met_transverse_momentum(nt).rho) / nt.pt for nt in WH_ntuples_dict.values()], axis = 0)
WH_muon_pt_balance = np.concatenate([(nt.pt - get_muon_met_transverse_momentum(nt).rho) / nt.pt for nt in WH_ntuples_dict.values()], axis = 0)
Min_DPhi_W_Jets = np.concatenate([nt.Min_DPhi_W_Jets for nt in WH_ntuples_dict.values()], axis = 0)

delta_phi_diphoton_W_lepton = np.where(is_electron_event, delta_phi_diphoton_W_electron, delta_phi_diphoton_W_muon)
WH_lepton_pt_balance = np.where(is_electron_event, WH_electron_pt_balance, WH_muon_pt_balance)

# WH_combined array with electron and muon concatenated into lepton.
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
print("===========================================================================")
print(WH_combined)
print(type(WH_combined))
print(WH_combined.shape)
print("===========================================================================")
print("\n")

WH_combined_transposed = WH_combined.T
print("output_array:")
print("===========================================================================")
print(WH_combined_transposed)
print("===========================================================================")
print(WH_combined_transposed.shape)
print("===========================================================================")
print(weight)
print("===========================================================================")
print(weight.shape)
################################################################################################################################
########## Save the feature arrays and weight array to npy files separately ##########
################################################################################################################################
output_dir = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/features_trial4_include_weights"
os.makedirs(output_dir, exist_ok = True)

np.save(file = os.path.join(output_dir, f"{args.output_filename}.npy"), arr = WH_combined_transposed)
np.save(file = os.path.join(output_dir, f"{args.output_filename}_weights.npy"), arr = weight)