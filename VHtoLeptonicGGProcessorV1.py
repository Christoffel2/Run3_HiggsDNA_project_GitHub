from higgs_dna.workflows.base import HggBaseProcessor

from higgs_dna.tools.chained_quantile import ChainedQuantileRegression
from higgs_dna.tools.diphoton_mva import calculate_diphoton_mva
from higgs_dna.tools.xgb_loader import load_bdt
from higgs_dna.tools.photonid_mva import calculate_photonid_mva, load_photonid_mva
from higgs_dna.tools.photonid_mva import calculate_photonid_mva_run3, load_photonid_mva_run3
from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.EELeak_region import veto_EEleak_flag
from higgs_dna.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from higgs_dna.tools.gen_helpers import get_fiducial_flag, get_genJets, get_higgs_gen_attributes
from higgs_dna.tools.sigma_m_tools import compute_sigma_m
from higgs_dna.tools.flow_corrections import calculate_flow_corrections
from higgs_dna.tools.mass_decorrelator import decorrelate_mass_resolution

from higgs_dna.selections.photon_selections import photon_preselection
from higgs_dna.selections.diphoton_selections import apply_fiducial_cut_det_level
from higgs_dna.selections.lepton_selections import select_electrons, select_muons
from higgs_dna.selections.jet_selections import select_jets, jetvetomap
from higgs_dna.selections.lumi_selections import select_lumis

from higgs_dna.utils.dumping_utils import diphoton_ak_array, dump_ak_array, diphoton_list_to_pandas, dump_pandas, get_obj_syst_dict

from higgs_dna.utils.misc_utils import choose_jet

from higgs_dna.metaconditions import photon_id_mva_weights
from higgs_dna.metaconditions import diphoton as diphoton_mva_dir

from higgs_dna.systematics import object_systematics as available_object_systematics
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections

import functools
import warnings
import awkward as ak
import logging
import numpy as np
import operator
import sys
import vector
from typing import Any, Dict, List, Optional
from coffea.nanoevents.methods import candidate
from coffea.analysis_tools import Weights
from copy import deepcopy

logger = logging.getLogger(__name__)

vector.register_awkward() 
# "register_awkward()" registers "awkward.Array" behaviors globally with "vector" so that we can identify certain awkward arrays as four vectors
# and have the operations overloaded as we expected.

class VHtoLeptonicGGProcessorV1(HggBaseProcessor):
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Dict[str, List[Any]] = None,
        corrections: Dict[str, List[Any]] = None,
        apply_trigger: bool = False,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        trigger_group = ".*DoubleEG.*",
        # "trigger_group" variable actually depends on what you enter in "--triggerGroup" argument in "runner_util.py".
        analysis = "mainAnalysis",
        # "analysis" variable is a key under "TriggerPaths/trigger_group/" in the metacondition file. 
        # For example, in "Era2022_v1.json", there are 2 trigger files under "TriggerPaths/trigger_group/mainAnalysis/".
        skipCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
        fiducialCuts: str = "classical",
        doDeco: bool = False,
        Smear_sigma_m: bool = False,
        doFlow_corrections: bool = False,
        output_format: str = "parquet"
    ):
        super().__init__(
            metaconditions,
            systematics = systematics,
            corrections = corrections,
            apply_trigger = apply_trigger,
            output_location = output_location,
            taggers = taggers,
            trigger_group = trigger_group,
            analysis = analysis,
            skipCQR = skipCQR,
            skipJetVetoMap = skipJetVetoMap,
            year = year,
            fiducialCuts = fiducialCuts,
            doDeco = doDeco,
            Smear_sigma_m = Smear_sigma_m,
            doFlow_corrections = doFlow_corrections,
            output_format = output_format
        ) # The values for these variables depends on what you enter in arguments in runner_utils.py.

    def process_extra(self, events):
        return events, {}
    
    def choose_leptons(self, leptonic_variables, n, fill_value):
        chosen_lepton_variables = leptonic_variables[ak.local_index(leptonic_variables, axis = 1) == n]
        chosen_lepton_variables = ak.pad_none(chosen_lepton_variables, 1)
        chosen_lepton_variables = ak.flatten(ak.fill_none(chosen_lepton_variables, fill_value))

        return chosen_lepton_variables

    def GetCategory(self, ievt):
        evt_nElectrons = self.nElectrons[ievt]
        evt_nMuons = self.nMuons[ievt]
        evt_nLeptons = evt_nElectrons + evt_nMuons

        if evt_nElectrons == 2 or evt_nMuons == 2:  # ZH-leptonic (same-flavor charged lepton pair)
            cat = 2
        elif evt_nLeptons == 1:  # WH-leptonic (single charged lepton)
            cat = 1
        else:  # VH-MET
            cat = 0

        return cat
              
    def process(self, events):
        # For details about where do "events" and "metadata" come from,
        # you can refer to "coffea backports-v0.7.x/coffea/processor/executor/Runner/_work_function()".
        dataset_name = events.metadata["dataset"]
        print(f"dataset_name is: {dataset_name}")

        """========== Data or Monte Carlo? =========="""
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"
        print(f"Data kind is: {self.data_kind}")

        """========== Here we start recording possible coffea accumulators, most likely histograms, could be counters, arrays, ... =========="""
        histos_etc = {}
        histos_etc[dataset_name] = {}
        if self.data_kind == "mc":
            histos_etc[dataset_name]["nTot"] = int(ak.num(events.genWeight, axis = 0))
            histos_etc[dataset_name]["nPos"] = int(ak.sum(events.genWeight > 0)) # ak.sum(events.genWeight > 0)" here gives you the number of genWeight values that are greater than 0.
            histos_etc[dataset_name]["nNeg"] = int(ak.sum(events.genWeight < 0)) # Same reasoning as "ak.sum(events.genWeight > 0)".
            histos_etc[dataset_name]["nEff"] = int((histos_etc[dataset_name]["nPos"] - histos_etc[dataset_name]["nNeg"]))
            histos_etc[dataset_name]["genWeightSum"] = float(ak.sum(events.genWeight)) # If the argument feed into ak.sum() is not mask, it returns the sum over all values in genweight array.
        else:
            histos_etc[dataset_name]["nTot"] = int(len(events))
            histos_etc[dataset_name]["nPos"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["nNeg"] = int(0)
            histos_etc[dataset_name]["nEff"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["genWeightSum"] = float(len(events))

        print(f"initial histos_etc is: {histos_etc}")

        if self.data_kind == "mc":
            raw_genWeights = events.genWeight
            sum_of_raw_genWeights = ak.sum(raw_genWeights, axis = 0) # This variable is used for weighting the histograms.
            print("=====================================================================")
            print(f"The raw genWeights of the MC sample: {raw_genWeights}")
            print(f"The sum of raw genWeights of the MC sample: {sum_of_raw_genWeights}")
            print("=====================================================================")

        """========== Lumi mask =========="""
        if self.data_kind == "data":
            try:
                lumimask = select_lumis(self.year[dataset_name][0], events, logger)
                events = events[lumimask]
            except:
                logger.info(f"[lumimask] Skip now! Unable to find year info of {dataset_name}")
        
        """
        ========== Apply jetvetomap: Only retain events that without any jets in the EE leakage region 
        (You can check this webpage for further info: https://cms-jerc.web.cern.ch/Recommendations/#jet-veto-maps) ==========
        """
        if not self.skipJetVetoMap:
            events = jetvetomap(events, logger, dataset_name, year = self.year[dataset_name][0])
        
        """========== metadata array to append to higgsdna output =========="""
        metadata = {}

        if self.data_kind == "mc":
            metadata["sum_genw_presel"] = str(ak.sum(events.genWeight)) # Add sum of gen weights before selections for normalisation in postprocessing.
        else:
            metadata["sum_genw_presel"] = "Data"

        print(f"initial metadata (metadata array to append to higgsdna output): {metadata}")
        print(metadata["sum_genw_presel"])

        """========== Apply filters and triggers =========="""
        events = self.apply_filters_and_triggers(events) # "apply_filters_and_triggers()" is a method defined in HggBaseProcessor in base.py.

        """========== Remove events affected by EcalBadCalibCrystal =========="""
        if self.data_kind == "data":
            events = remove_EcalBadCalibCrystal_events(events)

        """========== We need ScEta for corrections and systematics, it is present in NanoAODv13+ and can be calculated using PV for older versions. =========="""
        events.Photon = add_photon_SC_eta(events.Photon, events.PV)

        """========== Add veto EE leak branch for photons, could also be used for electrons. =========="""
        if ((self.year[dataset_name][0] == "2022EE") 
            or (self.year[dataset_name][0] == "2022postEE")
            ):
            events.Photon = veto_EEleak_flag(self, events.Photon)
            events.Electron = veto_EEleak_flag(self, events.Electron) # I add this line since my analysis is related to leptonic decay.

        """========== Read which systematics and corrections to process. =========="""
        try:
            correction_names = self.corrections[dataset_name]
            print(f"correction_names is: {correction_names}")

        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset_name]
            print(f"systematic_names is: {systematic_names}")

        except KeyError:
            systematic_names = []
        
        """
        ========== If Smear_sigma_m == True and no Smearing correction in .json for MC, it will throws an error. 
        Since the pt scpectrum need to be smeared in order to properly calculate the smeared sigma_m_m. ==========
        """
        if ((self.data_kind == "mc") 
            and (self.Smear_sigma_m) 
            and ("Smearing" not in correction_names)
            ):
            warnings.warn("Smearing should be specified in the corrections field in .json in order to smear the mass!")
            sys.exit(0) # "exit(0)" means a clean exit without any errors / problems

        """
        ========== Since we are now applying smearing term to the sigma_m_m, I added this portion of code
        specially for the estimation of smearing terms for the data events [data pt/energy] that are not smeared! ==========
        """
        if ((self.data_kind == "data") 
            and (self.Smear_sigma_m)
            ):
            correction_name = "Smearing"
            logger.info(f"\nApplying correction {correction_name} to dataset {dataset_name}\n")
            varying_function = available_object_corrections[correction_name]
            print(f"varying_function (data & (Smear_sigma_m == True)) is: {varying_function}")

            events = varying_function(events = events, year = self.year[dataset_name][0])
            print(f"events of (events = varying_function() (data & (Smear_sigma_m == True))) is: {events}")

        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                
                logger.info(f"Applying correction {correction_name} to dataset {dataset_name}")
                varying_function = available_object_corrections[correction_name]

                print(f"varying_function is: {varying_function}")

                events = varying_function(events = events, year = self.year[dataset_name][0])

                print(f"events of (events = varying_function()) is: {events}")
            
            elif correction_name in available_weight_corrections:
                # event weight corrections will be applied after photon preselection / application of further taggers.
                continue

            else:
                # may want to throw an error instead, needs to be discussed.
                warnings.warn(f"Could not process correction {correction_name}.")
                continue

        """========== Everything above seems fine for now, focus on editing the codes below (8/27) =========="""
        original_photons = events.Photon
        # Since jets object seems to be important in the reconstruction and categorisation of VH MET category (missing transverse energy), I add them back (8/27).
        original_jets = events.Jet

        """========== Systematic object variations =========="""
        for systematic_name in systematic_names:
            if systematic_name in available_object_systematics.keys():
                systematic_dct = available_object_systematics[systematic_name]
                print(f"systematic_dct is: {systematic_dct}")

                if systematic_dct["object"] == "Photon":
                    logger.info(f"Adding systematic {systematic_name} to photons collections of dataset {dataset_name}")
                    original_photons.add_systematic(
                        name = systematic_name, # name of the systematic variation / uncertainty source.
                        kind = systematic_dct["args"]["kind"], # the name of the kind of systematic variation.
                        what = systematic_dct["args"]["what"], # "what" gets varied.
                        varying_function = functools.partial(
                            systematic_dct["args"]["varying_function"],
                            events = events,
                            year = self.year[dataset_name][0],
                        ) # a function that describes how 'what' is varied, it must close over all non-event-data arguments.
                    ) # "add_systematic" is a function of coffea/nanoevents/methods/base.py
                # elif systematic_dct["object"] == "Jet":
                #     logger.info(f"Adding systematic {systematic_name} to jets collections of dataset {dataset_name}")
                #     original_jets.add_systematic(
                #         name = systematic_name,
                #         kind = systematic_dct["args"]["kind"],
                #         what = systematic_dct["args"]["what"],
                #         varying_function = functools.partial(
                #             systematic_dct["args"]["varying_function"],
                #             events = events,
                #             year = self.year[dataset_name][0],
                #         )
                #     )

            elif systematic_name in available_weight_systematics:
                continue
            else:
                warnings.warn(f"Could not process systematic variation {systematic_name}.")
                continue

        """========== Computing the normalizing flow correction =========="""
        if ((self.data_kind == "mc") and (self.doFlow_corrections)):
            # Applying the flow corrections to all photons before preselection. (Why?)
            counts = ak.num(original_photons)
            correctied_inputs, var_list = calculate_flow_corrections(original_photons, events, self.meta["flashggPhotons"]["flow_inputs"], self.meta["flashggPhotons"]["Isolation_transform_order"], year = self.year[dataset_name][0])

            # Store the raw NanoAOD value and update photon ID MVA value for preselection
            original_photons["mvaID_run3"] = ak.unflatten(self.add_photonid_mva_run3(original_photons, events), counts)
            original_photons["mvaID_nano"] = original_photons["mvaID"]

            # Store the raw values of the inputs and update the input values with the corrections since some variables are used in the preselection.
            for i in range(len(var_list)):
                original_photons["raw_" + str(var_list[i])] = original_photons[str(var_list[i])]
                original_photons[str(var_list[i])] = ak.unflatten(correctied_inputs[:,i], counts)

            original_photons["mvaID"] = ak.unflatten(self.add_photonid_mva_run3(original_photons, events), counts) # This line seems weird, maybe it is not needed.
        
        """========== Applying systematic variations =========="""
        photons_dct = {}
        photons_dct["nominal"] = original_photons
        logger.debug(original_photons.systematics.fields)
        for systematic in original_photons.systematics.fields:
            for variation in original_photons.systematics[systematic].fields:
                # deepcopy to allow for independent calculations on photon variables with CQR
                photons_dct[f"{systematic}_{variation}"] = deepcopy(original_photons.systematics[systematic][variation])

        print(f"photons_dct is: {photons_dct}")

        jets_dct = {}
        jets_dct["nominal"] = original_jets
        logger.debug(original_jets.systematics.fields)
        # for systematic in original_jets.systematics.systematics.fields:
        #     for variation in original_jets.systematics[systematic].fields:
        #         jets_dct[f"{systematic}_{variation}"] = deepcopy(original_jets.systematics[systematic][variation])
        
        print(f"jets_dct is: {jets_dct}")

        for variation, photons in photons_dct.items():
            for jet_variation, Jets in jets_dct.items():
                # make sure no duplicate executions
                if ((variation == "nominal") or (jet_variation == "nominal")):
                    if ((variation != "nominal") and (jet_variation != "nominal")):
                        continue
                    do_variation = "nominal"
                    if not ((variation == "nominal") and (jet_variation == "nominal")):
                        do_variation = variation if (variation != "nominal") else jet_variation
                    
                    logger.debug("Variation: {}".format(do_variation))
                    
                    if self.chained_quantile is not None:
                        photons = self.chained_quantile.apply(photons, events)
                    # recompute photonid_mva on the fly. (Why?)
                    if self.photonid_mva_EB and self.photonid_mva_EE:
                        photons = self.add_photonid_mva(photons, events)

                    """========== Photon preselection and turn the diphotons into candidates with four momenta and such =========="""
                    photons = photon_preselection(self, photons, events, year = self.year[dataset_name][0])
                    photons["charge"] = ak.zeros_like(photons.pt)
                    # photons["mvaID_score"] = photons.mvaID
                    # print(f"MVAid score of photons: {photons.mvaID_score}") # I commented out these two lines since they are automatically included in the Ntuples.

                    diphotons = ak.combinations(photons, 2, fields = ["pho_lead", "pho_sublead"])
                    # leading photons' pt > 35. (below) and subleading photons' pt > 25. (Contained in "photon_preselection()").
                    diphotons = diphotons[diphotons["pho_lead"].pt > self.min_pt_lead_photon]

                    # # Adding the leading- and subleading- photons' pt in each event, sort these value from big to small.
                    # # This will be used the moment we want to pick out the diphoton pair having the largest scalar pt sum in each event.
                    # lead_and_sublead_pt_sum = diphotons.pho_lead.pt + diphotons.pho_sublead.pt
                    # diphotons = diphotons[ak.argsort(lead_and_sublead_pt_sum, axis = 1, ascending = False)]
                    
                    diphoton_4momentum = diphotons["pho_lead"] + diphotons["pho_sublead"]
                    diphotons["pt"] = diphoton_4momentum.pt
                    diphotons["eta"] = diphoton_4momentum.eta
                    diphotons["phi"] = diphoton_4momentum.phi
                    diphotons["mass"] = diphoton_4momentum.mass
                    diphotons["charge"] = diphoton_4momentum.charge
                    diphotons_pz = diphoton_4momentum.z
                    diphotons_e = diphoton_4momentum.energy
                    diphotons["rapidity"] = 0.5 * np.log((diphotons_e + diphotons_pz)/(diphotons_e - diphotons_pz))
                    diphotons = ak.with_name(diphotons, "PtEtaPhiMCandidate")

                    if self.data_kind == "mc":
                        # diphotons["raw_genWeight"] = raw_genWeights
                        diphotons["sum_of_raw_genWeights"] = sum_of_raw_genWeights
                        diphotons["number_of_pileup"] = events.Pileup.nPU
                        print("=====================================================================")
                        # print(f"diphoton raw genweights is: {diphotons.raw_genWeight}")
                        print(f"diphoton sum of raw genweights is: {diphotons.sum_of_raw_genWeights}")
                        print("\n")
                        print(f"diphoton number of pileup field is: {diphotons.number_of_pileup}")
                        print("=====================================================================")
                        print("\n")

                    print("=====================================================================")
                    print(f"diphoton pt is: {diphotons.pt}")
                    print(f"diphoton eta is: {diphotons.eta}")
                    print(f"diphoton phi is: {diphotons.phi}")
                    print(f"diphoton mass is: {diphotons.mass}")
                    print(f"diphoton charge is: {diphotons.charge}")
                    print(f"The number of diphoton events: {ak.num(diphotons, axis = 0)}")
                    print("=====================================================================")
                    print("\n")

                    # Sort diphotons by pt
                    diphotons = diphotons[ak.argsort(diphotons.pt, axis = 1, ascending = False)]
                    
                    # Determine if event passes fiducial Hgg cuts at detector-level
                    diphotons = apply_fiducial_cut_det_level(self, diphotons)

                    # Baseline modifications to diphotons
                    if self.diphoton_mva is not None:
                        diphotons = self.add_diphoton_mva(diphotons, events)

                    # Workflow specific processing (Not sure about its role for now (9/9))
                    events, process_extra = self.process_extra(events)
                    histos_etc.update(process_extra)

                    # Jet variables
                    jets = ak.zip(
                        {
                            "pt" : Jets.pt,
                            "eta" : Jets.eta,
                            "phi" : Jets.phi,
                            "mass" : Jets.mass,
                            "charge" : ak.zeros_like(Jets.pt),
                            "jetId" : Jets.jetId,
                            "hadronFlavor" : Jets.hadronFlavour if self.data_kind == "mc" else ak.zeros_like(Jets.pt),
                            "btagPNetB" : Jets.btagPNetB,
                            "btagDeepFlavB" : Jets.btagDeepFlavB,
                            "btagRobustParTAK4B" : Jets.btagRobustParTAK4B,
                        }, with_name = "PtEtaPhiMCandidate"
                    )

                    # Electron variables
                    electrons = ak.zip(
                        {
                            "pt" : events.Electron.pt,
                            "eta" : events.Electron.eta,
                            "phi" : events.Electron.phi,
                            "mass" : events.Electron.mass,
                            "charge" : events.Electron.charge,
                            "cutBased" : events.Electron.cutBased,
                            "mvaIso_WP90" : events.Electron.mvaIso_WP90,
                            "mvaIso_WP80" : events.Electron.mvaIso_WP80,
                            "leptonID" : events.Electron.mvaIso,
                            "leptonFlavour" : ak.full_like(events.Electron.pt, 0),
                        }, with_name = "PtEtaPhiMCandidate"
                    )

                    # Muon variables
                    muons = ak.zip(
                        {
                            "pt" : events.Muon.pt,
                            "eta": events.Muon.eta,
                            "phi": events.Muon.phi,
                            "mass": events.Muon.mass,
                            "charge": events.Muon.charge,
                            "tightId": events.Muon.tightId,
                            "mediumId": events.Muon.mediumId,
                            "looseId": events.Muon.looseId,
                            "isGlobal": events.Muon.isGlobal,
                            "pfIsoId" : events.Muon.pfIsoId,
                            "leptonFlavour" : ak.full_like(events.Muon.pt, 1),
                            "leptonID" : events.Muon.mvaMuID,
                        }, with_name = "PtEtaPhiMCandidate"
                    )

                    # Lepton cleaning and sorting --- electrons and muons.
                    electrons = electrons[select_electrons(self, electrons, diphotons)]
                    electrons = electrons[ak.argsort(electrons.pt, axis = 1, ascending = False)]
                    muons = muons[select_muons(self, muons, diphotons)]
                    muons = muons[ak.argsort(muons.pt, axis = 1, ascending = False)]

                    # Categorize events based on lepton flavor and count, and save this information in the diphotons' fields.
                    nElectrons = ak.num(electrons, axis = 1)
                    nMuons = ak.num(muons, axis = 1)
                    nLeptons = np.add(nElectrons, nMuons)
                    nDiphotons = ak.num(diphotons.pt, axis = 1)

                    self.nElectrons = nElectrons
                    self.nMuons = nMuons
                    self.nLeptons = nLeptons

                    ievt_by_diphotons = ak.flatten(
                        ak.Array([nDipho * [evt_i] for evt_i, nDipho in enumerate(nDiphotons)])
                    )
                    cat_vals = ak.Array(map(self.GetCategory, ievt_by_diphotons))
                    cat_vals = ak.unflatten(cat_vals, nDiphotons)
                    diphotons["event_category"] = cat_vals

                    # MET (Missing Transverse Energy) variables (For neutrinos)
                    met = ak.zip(
                        {
                            "pt" : events.MET.pt,
                            "phi" : events.MET.phi,
                            "significance" : events.MET.significance,
                            "sumEt" : events.MET.sumEt,
                        }, with_name = "PtEtaPhiMCandidate"
                    )

                    # MET selection
                    met_selection = met.pt > 75.
                    met = ak.mask(met, met_selection)

                    # Add lepton variables to diphotons. (This part is originally written by Tom Runting in stxs.py of HiggsDNA.)
                    lepton_indices = [0,1]
                    diphotons["n_electrons"] = nElectrons
                    diphotons["n_muons"] = nMuons

                    for i in lepton_indices:
                        # electrons' kinematic variables.
                        diphotons[f"electron{i}_pt"] = self.choose_leptons(electrons.pt, i, -999.0)
                        diphotons[f"electron{i}_eta"] = self.choose_leptons(electrons.eta, i, -999.0)
                        diphotons[f"electron{i}_phi"] = self.choose_leptons(electrons.phi, i, -999.0)
                        diphotons[f"electron{i}_mass"] = self.choose_leptons(electrons.mass, i, -999.0)
                        diphotons[f"electron{i}_charge"] = self.choose_leptons(electrons.charge, i, -999.0)
                        diphotons[f"electron{i}_leptonFlavour"] = self.choose_leptons(electrons.leptonFlavour, i, -999.0)
                        diphotons[f"electron{i}_id"] = self.choose_leptons(electrons.leptonID, i, -999.0)
                        # muons' kinematic variables.
                        diphotons[f"muon{i}_pt"] = self.choose_leptons(muons.pt, i, -999.0)
                        diphotons[f"muon{i}_eta"] = self.choose_leptons(muons.eta, i, -999.0)
                        diphotons[f"muon{i}_phi"] = self.choose_leptons(muons.phi, i, -999.0)
                        diphotons[f"muon{i}_mass"] = self.choose_leptons(muons.mass, i, -999.0)
                        diphotons[f"muon{i}_charge"] = self.choose_leptons(muons.charge, i, -999.0)
                        diphotons[f"muon{i}_leptonFlavour"] = self.choose_leptons(muons.leptonFlavour, i, -999.0)
                        diphotons[f"muon{i}_id"] = self.choose_leptons(muons.leptonID, i, -999.0)
                    
                    print("=====================================================================")
                    print(f"diphotons' fields after adding electrons and muons: {diphotons.fields}")
                    print("\n")
                    print(f"The number of electron events: {ak.num(diphotons.electron0_pt, axis = 0)}")
                    print("=====================================================================")
                    print("\n")

                    diphotons["MET_pt"] = met.pt
                    diphotons["MET_phi"] = met.phi
                    diphotons["MET_sumEt"] = met.sumEt
                    
                    print("=====================================================================")
                    print(f"diphotons' fields after adding MET's variables: {diphotons.fields}")
                    print("\n")
                    print(f"The number of MET events: {ak.num(diphotons.MET_pt, axis = 0)}")
                    print("=====================================================================")
                    print("\n")

                    # # PUPPI MET objects (Not sure about its role for now (8/27))
                    # puppiMET = events.PuppiMET
                    # puppiMET = ak.with_name(puppiMET, "PtEtaPhiMCandidate")

                    # Jet selection
                    jets = jets[select_jets(self, jets, diphotons, muons, electrons)]
                    jets = jets[ak.argsort(jets.pt, axis = 1, ascending = False)]

                    events["sel_jets"] = jets
                    n_jets = ak.num(jets, axis = 1)

                    # Add jets. (This part is originally written by Tom Runting in stxs.py of HiggsDNA.)
                    jet_indices = [0, 1, 2, 3]
                    jet_collection = {}
                    for i in jet_indices:
                        jet_collection[f"Jet{i}_pt"] = choose_jet(jets.pt, i, -999.0)
                        jet_collection[f"Jet{i}_eta"] = choose_jet(jets.eta, i, -999.0)
                        jet_collection[f"Jet{i}_phi"] = choose_jet(jets.phi, i, -999.0)
                        jet_collection[f"Jet{i}_mass"] = choose_jet(jets.mass, i, -999.0)
                        jet_collection[f"Jet{i}_charge"] = choose_jet(jets.charge, i, -999.0)
                        jet_collection[f"Jet{i}_btagPNetB"] = choose_jet(jets.btagPNetB, i, -999.0)
                        jet_collection[f"Jet{i}_btagDeepFlavB"] = choose_jet(jets.btagDeepFlavB, i, -999.0)
                        jet_collection[f"Jet{i}_btagRobustParTAK4B"] = choose_jet(jets.btagRobustParTAK4B, i, -999.0)

                    # Add Ht (scalar sum of jet Et). (This part is originally written by Tom Runting in stxs.py of HiggsDNA.)
                    # Do I need this variable in my analysis? --> Only need jets_Et.
                    jets_Et = np.sqrt(jets.pt**2 + jets.mass**2)
                    jets_Ht = ak.sum(jets_Et, axis = 1)
                    jets_Ht = ak.fill_none(jets_Ht, -999.0)

                    # Originally I only added leading-jets variables to diphotons' fields since leading-jet variables are used in my analysis as a discriminator,
                    # however, to avoid missing something about jets in future, I decide to add all jets' variables to diphotons' fields.
                    # (If we don't consider jets, ttH-leptonic and VH-leptonic are nearly the same.)
                    for i in jet_indices:
                        diphotons[f"Jet{i}_pt"] = jet_collection[f"Jet{i}_pt"]
                        diphotons[f"Jet{i}_eta"] = jet_collection[f"Jet{i}_eta"]
                        diphotons[f"Jet{i}_phi"] = jet_collection[f"Jet{i}_phi"]
                        diphotons[f"Jet{i}_mass"] = jet_collection[f"Jet{i}_mass"]
                        diphotons[f"Jet{i}_charge"] = jet_collection[f"Jet{i}_charge"]
                        diphotons[f"Jet{i}_btagPNetB"] = jet_collection[f"Jet{i}_btagPNetB"]
                        diphotons[f"Jet{i}_btagDeepFlavB"] = jet_collection[f"Jet{i}_btagDeepFlavB"]
                        diphotons[f"Jet{i}_btagRobustParTAK4B"] = jet_collection[f"Jet{i}_btagRobustParTAK4B"]
                    diphotons["n_jets"] = n_jets
                    print("=====================================================================")
                    print(f"diphotons' fields after adding ''jets' variables'': {diphotons.fields}")
                    print("=====================================================================")
                    print("\n")

                    # Calculate the minimum of the azimuthal angle between the transverse momentum of MET and the transverse momentum of one of the jet.
                    ## Broadcast the phi array of MET to the same shape as that of jets.
                    MET_phi_broadcasted = ak.broadcast_arrays(met.phi, jets.phi)[0]
                    DPhi_MET_Jets = np.abs(MET_phi_broadcasted - jets.phi)
                    DPhi_MET_Jets = ak.where(DPhi_MET_Jets > np.pi, 2 * np.pi - DPhi_MET_Jets, DPhi_MET_Jets)
                    Min_DPhi_MET_Jets = ak.min(DPhi_MET_Jets, axis = 1)
                    Min_DPhi_MET_Jets = ak.fill_none(Min_DPhi_MET_Jets, -999.0, axis = 0)
                    diphotons["Min_DPhi_MET_Jets"] = Min_DPhi_MET_Jets
                    print(f"Min_DPhi_MET_Jets is: {diphotons.Min_DPhi_MET_Jets}")
                    print("\n")

                    # Calculate the minimum of the azimuthal angle between the transverse momentum of W boson and the transverse momentum of each jet,
                    # where W boson decays to a lepton and a neutrino.
                    leptons = ak.concatenate([electrons, muons], axis = 1)
                    leptons = ak.mask(leptons, ak.num(leptons, axis = 1) == 1)
                    met = ak.mask(met, ak.num(leptons, axis = 1) == 1)
                    jets = ak.mask(jets, ak.num(leptons, axis = 1) == 1)
                    leading_lepton = ak.firsts(leptons)  # Now we are sure there's exactly one lepton per event
                    # Construct the W boson transverse momentum as the sum of lepton and MET
                    W_px = leading_lepton.pt * np.cos(leading_lepton.phi) + met.pt * np.cos(met.phi)
                    W_py = leading_lepton.pt * np.sin(leading_lepton.phi) + met.pt * np.sin(met.phi)
                    W_phi = np.arctan2(W_py, W_px)  # Azimuthal angle of the W boson
                    W_phi = ak.fill_none(W_phi, -999.0)
                    print(f"W_phi is {W_phi}")
                    print("W_phi:", ak.type(W_phi))
                    print("\n")

                    jets_phi = jets.phi
                    print(f"jets_phi is {jets_phi}")
                    print("jets_phi:", ak.type(jets_phi))
                    print("\n")
                    W_phi_jagged, jets_phi = ak.broadcast_arrays(W_phi, jets_phi) # Broadcast w_phi to match jets_phi's jagged structure
                    print("W_phi_jagged:", ak.type(W_phi_jagged))
                    print("\n")
                    print("jets_phi:", ak.type(jets_phi))
                    print("\n")
                    DPhi_W_Jets = np.abs(W_phi_jagged - jets_phi)
                    print(f"DPhi_W_Jets array before ak.where() is: {DPhi_W_Jets}")
                    print("\n")
                    print(f"The number of DPhi_W_Jets array in axis = 0 before ak.where() is: {ak.num(DPhi_W_Jets, axis = 0)}")
                    print("\n")
                    print(f"The number of DPhi_W_Jets array in axis = 1 before ak.where() is: {ak.num(DPhi_W_Jets, axis = 1)}")
                    print("\n")
                    DPhi_W_Jets = np.where(DPhi_W_Jets > np.pi, 2 * np.pi - DPhi_W_Jets, DPhi_W_Jets)
                    print(f"DPhi_W_Jets array after ak.where() is: {DPhi_W_Jets}")
                    print("\n")
                    print(f"The number of DPhi_W_Jets array in axis = 0 after ak.where() is: {ak.num(DPhi_W_Jets, axis = 0)}")
                    print("\n")
                    print(f"The number of DPhi_W_Jets array in axis = 1 after ak.where() is: {ak.num(DPhi_W_Jets, axis = 1)}")
                    print("\n")
                    Min_DPhi_W_Jets = ak.min(DPhi_W_Jets, axis = 1)
                    print(f"Min_DPhi_W_Jets is: {Min_DPhi_W_Jets}")
                    print("\n")
                    print(f"The number of Min_DPhi_W_Jets in axis = 0 is: {ak.num(Min_DPhi_W_Jets, axis = 0)}")
                    print("\n")
                    diphotons["Min_DPhi_W_Jets"] = Min_DPhi_W_Jets
                    print("=====================================================================")
                    print(f"diphotons' fields after adding ''Min_DPhi_MET_Jets'' and ''Min_DPhi_W_Jets'' are: {diphotons.fields}")
                    print("=====================================================================")

                    """
                    ========== Run taggers on the events list with added diphotons. The Shape here is ensured to be broadcastable. ==========
                    """
                    for tagger in self.taggers:
                        (diphotons["_".join([tagger.name, str(tagger.priority)])], tagger_extra,) = tagger(events, diphotons)
                        histos_etc.update(tagger_extra)

                    """
                    ========== If there are taggers to run, arbitrate by them first; Deal with order of tagger priorities; 
                    Turn from diphoton jagged array to whether or not an event was selected. ==========
                    """
                    if len(self.taggers):
                        counts = ak.num(diphotons.pt, axis = 1)
                        flat_tags = np.stack(
                            (
                                ak.flatten(diphotons["_".join([tagger.name, str(tagger.priority)])]) for tagger in self.taggers
                            ), axis = 1,
                        )
                        tags = ak.from_regular(ak.unflatten(flat_tags, counts), axis = 2)
                        winner = ak.min(tags[tags != 0], axis = 2)
                        diphotons["best_tag"] = winner

                        #Lowest priority is most important (ascending sort); Leave in order of diphoton pt in case of ties (stable sort).
                        sorted = ak.argsort(diphotons.best_tag, stable = True)
                        diphotons = diphotons[sorted]
                    
                    """========== ? =========="""
                    diphotons = ak.firsts(diphotons)
                    events[f"diphotons_{do_variation}"] = diphotons # Set diphotons as part of the event record.
                    diphotons["events"] = events.event
                    diphotons["lumi"] = events.luminosityBlock
                    diphotons["run"] = events.run
                    diphotons["nPV"] = events.PV.npvs # nPV for validation of pileup reweighting.
                    diphotons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll

                    if self.data_kind == "mc":
                        diphotons["genWeight"] = events.genWeight
                        diphotons["dZ"] = (events.GenVtx.z - events.PV.z)
                        # Necessary for differential cross section measurements in final fits ("truth" variables).
                        diphotons["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                        diphotons["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                        diphotons["HTXS_njets30"] = events.HTXS.njets30
                        diphotons["HTXS_stage_0"] = events.HTXS.stage_0 # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final states.
                    else:
                        diphotons["dZ"] = ak.zeros_like(events.PV.z) # Fill zeros for data because there is no GenVtx for data.

                    """
                    ========== Drop events without a preselected diphoton candidate and 
                    drop events without a tag if there are tags. ==========
                    """
                    if len(self.taggers):
                        selection_mask = ~(
                            ak.is_none(diphotons) | ak.is_none(diphotons.best_tag)
                        )
                        diphotons = diphotons[selection_mask]
                    else:
                        selection_mask = ~ak.is_none(diphotons)
                        diphotons = diphotons[selection_mask]
                    
                    """========== Return if there is no surviving events =========="""
                    if len(diphotons) == 0:
                        logger.debug("No surviving events in this run, return now!")
                        return histos_etc
                    
                    if self.data_kind == "mc":
                        # Initiate weight container here after selection, since event selection cannot easily be applied to weight container afterwards.
                        event_weights = Weights(size = len(events[selection_mask]))

                        # Corrections to event weights:
                        for correction_name in correction_names:
                            if correction_name in available_weight_corrections:
                                logger.info(f"Adding correction {correction_name} to weight collection of dataset {dataset_name}")
                                varying_function = available_weight_corrections[correction_name]
                                event_weights = varying_function(
                                    events = events[selection_mask],
                                    photons = events[f"diphotons_{do_variation}"][selection_mask],
                                    weights = event_weights,
                                    dataset_name = dataset_name,
                                    year = self.year[dataset_name][0],
                                )
                
                        # Systematic variations of event weights go to nominal output dataframe
                        if do_variation == "nominal":
                            for systematic_name in systematic_names:
                                if systematic_name in available_weight_systematics:
                                    logger.info(f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}")

                                    if systematic_name == "LHEScale":
                                        if hasattr(events, "LHEScaleWeight"):
                                            diphotons["nweight_LHEScale"] = ak.num(events.LHEScaleWeight[selection_mask], axis = 1,)
                                            diphotons["weight_LHEScale"] = events.LHEScaleWeight[selection_mask]
                                        else:
                                            logger.info(f"No {systematic_name} Weights in dataset {dataset_name}")

                                    elif systematic_name == "LHEPdf":
                                        if hasattr(events, "LHEPdfWeight"):
                                            diphotons["nweight_LHEPdf"] = (ak.num(events.LHEPdfWeight[selection_mask], axis = 1,) - 2) # Two AlphaS weighs are removed.
                                            diphotons["weight_LHEPdf"] = events.LHEPdfWeight[selection_mask][:,:-2]
                                        else:
                                            logger.info(f"No {systematic_name} Weights in dataset {dataset_name}")

                                    else:
                                        varying_function = available_weight_systematics[systematic_name]
                                        event_weights = varying_function(
                                            events = events[selection_mask],
                                            photons = events[f"diphotons_{do_variation}"][selection_mask],
                                            weights = event_weights,
                                            dataset_name = dataset_name,
                                            year = self.year[dataset_name][0],
                                        )
                        diphotons["weight"] = event_weights.weight()
                        diphotons["weight_central"] = event_weights.weight()
                        

                        # Store variations with respect to central weight.
                        if do_variation == "nominal":
                            if len(event_weights.variations):
                                logger.info("Adding systematic weight variations to nominal output file.")
                            
                            for modifier in event_weights.variations:
                                diphotons["weight" + modifier] = event_weights.weight(modifier = modifier)

                        # # Multiply weight by genWeight for normalisation in post-processing chain.
                        # event_weights._weight = ((events["genWeight"][selection_mask]) * (diphotons["weight_central"]))
                        # diphotons["weight"] = event_weights.weight()

                    else:
                        # Add weight variables (=1) for data for consistent datasets.
                        diphotons["weight_central"] = ak.ones_like(diphotons["events"])
                        diphotons["weight"] = ak.ones_like(diphotons["events"])

                    # Compute and store the different variations of sigma_m_over_m.
                    diphotons = compute_sigma_m(diphotons, processor = "base", flow_corrections = self.doFlow_corrections, smear = self.Smear_sigma_m)

                    # Decorrelating the mass resolution. (Still need to supress the decorrelator noises)
                    if self.doDeco:
                        # Decorrelate nominal sigma_m_over_m.
                        diphotons["sigma_m_over_m_nominal_decorr"] = decorrelate_mass_resolution(diphotons, type="nominal", year=self.year[dataset_name][0])

                        # Decorrelate smeared nominal sigma_m_over_m.
                        if (self.Smear_sigma_m):
                            diphotons["sigma_m_over_m_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="smeared", year=self.year[dataset_name][0])

                        # Decorrelate flow corrected sigma_m_over_m.
                        if (self.doFlow_corrections):
                            diphotons["sigma_m_over_m_corr_decorr"] = decorrelate_mass_resolution(diphotons, type="corr", year=self.year[dataset_name][0])

                        # Decorrelate flow corrected, smeared sigma_m_over_m.
                        if ((self.doFlow_corrections) and (self.Smear_sigma_m)):
                            diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0])
                    if self.output_location is not None:
                        if self.output_format == "root":
                            df = diphoton_list_to_pandas(self, diphotons)
                        else:
                            akarr = diphoton_ak_array(self, diphotons)

                            # Remove fixedGridRhoAll from photons to avoid having event-level info per photon
                            akarr = akarr[[field for field in akarr.fields if "lead_fixedGridRhoAll" not in field]]

                        fname = (events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".%s" % self.output_format)
                        fname = (fname.replace("%2F", "")).replace("%3B1", "")
                        subdirs = []

                        if "dataset" in events.metadata:
                            subdirs.append(events.metadata["dataset"])
                        
                        subdirs.append(do_variation)

                        if self.output_format == "root":
                            dump_pandas(self, df, fname, self.output_location, subdirs)
                        else:
                            dump_ak_array(self, akarr, fname, self.output_location, metadata, subdirs,)

        return histos_etc
    
    def process_extra(self, events):
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]):
        pass