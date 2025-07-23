#!/usr/bin/env python
''' Analysis script for standard plots
'''
#
# Standard imports and batch mode
#
import ROOT, os
ROOT.gROOT.SetBatch(True)
c1 = ROOT.TCanvas() # do this to avoid version conflict in png.h with keras import ...
c1.Draw()
c1.Print('delete.png')
import itertools
import copy
import array
import operator
from math                                import sqrt, cos, sin, pi, atan2, cosh, exp

# RootTools
from RootTools.core.standard             import *

# CorrelatorMtop
from CorrelatorMtop.Tools.user                 import plot_directory
from CorrelatorMtop.Tools.cutInterpreter       import cutInterpreter
from CorrelatorMtop.Tools.energyCorrelators    import getTriplets_pp_TLorentz_label
from CorrelatorMtop.Tools.helpers              import BreitWignerReweight, deltaPhi, deltaR, deltaRTLorentz, writeObjToFile
from CorrelatorMtop.samples.lumi_info          import lumi_info

import Analysis.Tools.syncer
import numpy as np

################################################################################
# Arguments
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',       action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--plot_directory', action='store', default='CorrelatorMtop_v1')
argParser.add_argument('--small',          action='store_true', help='Run only on a small subset of the data?', )
argParser.add_argument('--fillHists',      action='store_true', help='Store histograms?', )
argParser.add_argument('--selection_rec',  action='store', default='PFboosted-PFleppt')
argParser.add_argument('--selection_gen',  action='store', default='GENboosted')
argParser.add_argument('--era',            action='store', type=str, default="UL2018")
argParser.add_argument('--nJobs',          action='store', type=int, default=1, help="EFT interpolation order" )
argParser.add_argument('--job',            action='store', type=int, default=0, help="Run only jobs i" )
# argParser.add_argument('--split',          action='store_true', help='Split sample into data and simulation?')

args = argParser.parse_args()

################################################################################
# Logger
import CorrelatorMtop.Tools.logger as logger
import RootTools.core.logger as logger_rt
logger    = logger.get_logger(   args.logLevel, logFile = None)
logger_rt = logger_rt.get_logger(args.logLevel, logFile = None)

################################################################################
# Result dir
outdir = "/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/results/"

################################################################################
# Define the MC samples
from CorrelatorMtop.samples.UL2018_unprocessed import TTToSemiLeptonic as TTToSemiLeptonic_UL2018

if args.nJobs > 1:
    all_splits = TTToSemiLeptonic_UL2018.split(n=10, shuffle=False)
    mc = [TTToSemiLeptonic_UL2018.split(n=args.nJobs, shuffle=False)[args.job]]
else:
    mc = [TTToSemiLeptonic_UL2018]

for sample in mc:
    sample.scale = 1.
    if args.small:
        sample.normalization = 1.
        sample.reduceFiles( to = 1 )
        sample.scale /= sample.normalization

################################################################################
# Functions needed specifically for this analysis routine
def getHadronicTop(event):
    # First find the two tops
    foundTop = False
    foundATop = False
    for i in range(event.nGenPart):
        if foundTop and foundATop:
            break
        if event.GenPart_pdgId[i] == 6:
            top = ROOT.TLorentzVector()
            top.SetPtEtaPhiM(event.GenPart_pt[i],event.GenPart_eta[i],event.GenPart_phi[i],event.GenPart_m[i])
            foundTop = True
        elif event.GenPart_pdgId[i] == -6:
            atop = ROOT.TLorentzVector()
            atop.SetPtEtaPhiM(event.GenPart_pt[i],event.GenPart_eta[i],event.GenPart_phi[i],event.GenPart_m[i])
            foundATop = True
    # Now search for leptons
    # if grandmother is the top, the atop is hadronice and vice versa
    for i in range(event.nGenPart):
        if abs(event.GenPart_pdgId[i]) in [11, 13, 15]:
            if event.GenPart_grmompdgId[i] == 6:
                return atop
            elif event.GenPart_grmompdgId[i] == -6:
                return top
    return None

def getLeadingLepton(event, genpf_switch):
    # First find the lepton
    lepton = ROOT.TLorentzVector()
    if genpf_switch == "gen":
        for i in range(event.nGenPart):
            if abs(event.GenPart_pdgId[i]) in [11, 13, 15]:
                if abs(event.GenPart_grmompdgId[i]) == 6:
                    lepton.SetPtEtaPhiM(event.GenPart_pt[i],event.GenPart_eta[i],event.GenPart_phi[i],event.GenPart_m[i])
                    return lepton
    else:
        maxpt = 0
        for i in range(event.nElectron):
            if event.Electron_pt[i] > maxpt:
                maxpt = event.Electron_pt[i]
                lepton.SetPtEtaPhiM(event.Electron_pt[i],event.Electron_eta[i],event.Electron_phi[i], 0.0)
        for i in range(event.nMuon):
            if event.Muon_pt[i] > maxpt:
                maxpt = event.Muon_pt[i]
                lepton.SetPtEtaPhiM(event.Muon_pt[i],event.Muon_eta[i],event.Muon_phi[i], 0.0)
        return lepton
    return None

def getLeadingJet(event, genpf_switch):
    # First find the lepton
    jet = ROOT.TLorentzVector()
    id = None
    if genpf_switch == "gen":
        maxpt = 0
        for i in range(event.nGenJetAK8):
            if event.GenJetAK8_pt[i] > maxpt:
                maxpt = event.GenJetAK8_pt[i]
                jet.SetPtEtaPhiM(event.GenJetAK8_pt[i],event.GenJetAK8_eta[i],event.GenJetAK8_phi[i],event.GenJetAK8_mass[i])
                id = i
        return jet, id
    else:
        maxpt = 0
        for i in range(event.nPFJetAK8):
            if event.PFJetAK8_pt[i] > maxpt:
                maxpt = event.PFJetAK8_pt[i]
                jet.SetPtEtaPhiM(event.PFJetAK8_pt[i],event.PFJetAK8_eta[i],event.PFJetAK8_phi[i],event.PFJetAK8_mass[i])
                id = i
        return jet, id
    return None, None

def getClosestJetIdx(object, event, maxDR, genpf_switch):
    Njets = event.nGenJetAK8 if genpf_switch == "gen" else event.nPFJetAK8
    minDR = maxDR
    idx_match = None
    for i in range(Njets):
        jet = ROOT.TLorentzVector()
        if genpf_switch == "gen":
            jet.SetPtEtaPhiM(event.GenJetAK8_pt[i],event.GenJetAK8_eta[i],event.GenJetAK8_phi[i],event.GenJetAK8_mass[i])
        else:
            jet.SetPtEtaPhiM(event.PFJetAK8_pt[i],event.PFJetAK8_eta[i],event.PFJetAK8_phi[i],event.PFJetAK8_mass[i])
        if jet.DeltaR(object) < minDR:
            idx_match = i
            minDR = jet.DeltaR(object)
    return idx_match

def getChargedParticlesFromJet(event, jetidx, genrec, N_parts_max = None, pt_min = 0):
    chargedIds = [211, 13, 11, 1, 321, 2212, 3222, 3112, 3312, 3334]
    particles = []
    if genrec == "gen":
        for iPart in range(event.nGenJetAK8_cons):
            if event.GenJetAK8_cons_jetIndex[iPart] != jetidx:
                continue
            if abs(event.GenJetAK8_cons_pdgId[iPart]) not in chargedIds:
                continue
            particle = ROOT.TLorentzVector()
            particle.SetPtEtaPhiM(event.GenJetAK8_cons_pt[iPart],event.GenJetAK8_cons_eta[iPart],event.GenJetAK8_cons_phi[iPart],event.GenJetAK8_cons_mass[iPart])
            if particle.Pt() < pt_min:
                continue
            charge = 1 if event.GenJetAK8_cons_pdgId[iPart]>0 else -1
            particles.append( (particle, charge) )
    elif genrec == "rec":
        for iPart in range(event.nPFJetAK8_cons):
            if event.PFJetAK8_cons_jetIndex[iPart] != jetidx:
                continue
            if abs(event.PFJetAK8_cons_pdgId[iPart]) not in chargedIds:
                continue
            particle = ROOT.TLorentzVector()
            particle.SetPtEtaPhiM(event.PFJetAK8_cons_pt[iPart],event.PFJetAK8_cons_eta[iPart],event.PFJetAK8_cons_phi[iPart],event.PFJetAK8_cons_mass[iPart])
            if particle.Pt() < pt_min:
                continue
            charge = 1 if event.PFJetAK8_cons_pdgId[iPart]>0 else -1
            particles.append( (particle, charge) )
    # Sort list by pt and truncate at length N_parts_max
    particles_sorted = sorted(particles, key=lambda (particle, charge): particle.Pt(), reverse=True)
    if N_parts_max is not None:
        particles_sorted = particles_sorted[:N_parts_max]
    return particles_sorted

def passTripletSelection(triplet, ptjet, sel="top"):
    asymm_max_top = pow(172.5,2)/pow(ptjet,2)
    short_min_top = 0.1
    if sel=="top":
        if triplet[1] > asymm_max_top:
            return False
        if triplet[2] < short_min_top:
            return False
        return True
    if sel=="W":
        if triplet[1] > asymm_max_top:
            return False
        if triplet[2] > short_min_top:
            return False
        return True
    else:
        return False

def tag_objects(lst, tag):
    return [(obj, tag) for obj in lst]
################################################################################
# Define sequences
sequence       = []

def getConstituents( event, sample ):
    # Define all variables already here
    event.jetpt_gen = -1
    event.jetpt_rec = -1
    event.nGenParts = -1
    event.nPFParts = -1
    event.nGenAll = -1
    event.nGenMatched = -1
    event.matchingEffi = -1
    event.passSel = False
    event.zeta_gen = np.zeros( ( len([]), 3), dtype='f' )
    event.weight_gen = np.zeros( ( len([]), 1), dtype='f' )
    event.zeta_rec = np.zeros( ( len([]), 3), dtype='f' )
    event.weight_rec = np.zeros( ( len([]), 1), dtype='f' )
    event.zeta_gen_unmatched = np.zeros( ( len([]), 3), dtype='f' )
    event.weight_gen_unmatched = np.zeros( ( len([]), 1), dtype='f' )
    event.zeta_rec_unmatched = np.zeros( ( len([]), 3), dtype='f' )
    event.weight_rec_unmatched = np.zeros( ( len([]), 1), dtype='f' )

    # Get leading lepton and leading jet
    # Check that they separated and get jet constituent
    lep_gen = getLeadingLepton(event, "gen")
    jet_gen, jet_gen_id = getLeadingJet(event, "gen")
    if lep_gen is None or jet_gen is None:
        return
    if deltaRTLorentz(lep_gen, jet_gen) < 0.8:
        return
    constituent_gen = getChargedParticlesFromJet(event, jet_gen_id, "gen", N_parts_max = 25, pt_min = 5)
    event.nGenParts = len(constituent_gen)

    # Do the same for PF jets
    lep_rec = getLeadingLepton(event, "rec")
    jet_rec, jet_rec_id = getLeadingJet(event, "rec")
    if lep_rec is None or jet_rec is None:
        return
    if deltaRTLorentz(lep_rec, jet_rec) < 0.8:
        return
    constituent_rec = getChargedParticlesFromJet(event, jet_rec_id, "rec", N_parts_max = 25, pt_min = 5)
    event.nPFParts = len(constituent_rec)

    # Some numbers for histograms
    event.nGenParts_noCut = len(getChargedParticlesFromJet(event, jet_gen_id, "gen"))
    event.nGenParts_onlyPT = len(getChargedParticlesFromJet(event, jet_gen_id, "gen", pt_min = 5))
    event.nPFParts_noCut = len(getChargedParticlesFromJet(event, jet_rec_id, "rec"))
    event.nPFParts_onlyPT = len(getChargedParticlesFromJet(event, jet_rec_id, "rec", pt_min = 5))
    #######

    # Match constituents
    maxDR_part = 0.05
    genMatches = {}
    alreadyMatchedPF = []
    for i, (genPart, genCharge) in enumerate(constituent_gen):
        # Find all possible matching PF particle for a gen particle
        matches = []
        for j, (pfPart, pfCharge) in enumerate(constituent_rec):
            if j in alreadyMatchedPF:
                continue
            if genCharge == pfCharge and genPart.DeltaR(pfPart) < maxDR_part:
                matches.append(j)
        # If there are multiple matches, find the pf with same charge and closest in pt
        matchIDX = None
        if len(matches) == 0:
            matchIDX = None
        elif len(matches) == 1:
            matchIDX = matches[0]
        else:
            minPtDiff = 1000
            for idx in matches:
                PtDiff = abs(genPart.Pt()-constituent_rec[idx][0].Pt())
                if PtDiff < minPtDiff:
                    minPtDiff = PtDiff
                    matchIDX = idx
        genMatches[i] = matchIDX
        alreadyMatchedPF.append(matchIDX)

    constituent_gen_matched = []
    constituent_rec_matched = []
    constituent_gen_unmatched = []
    constituent_rec_unmatched = []
    for i, (genPart, genCharge) in enumerate(constituent_gen):
        if genMatches[i] is not None:
            # save matched
            constituent_gen_matched.append(constituent_gen[i][0])
            constituent_rec_matched.append(constituent_rec[genMatches[i]][0])
        else:
            # save gen particles without match
            constituent_gen_unmatched.append(constituent_gen[i][0])
    # now rec particles without match
    for i, (pfPart, pfCharge) in enumerate(constituent_rec):
        if i not in alreadyMatchedPF:
            constituent_rec_unmatched.append(constituent_rec[i][0])




    event.jetpt_gen = jet_gen.Pt()
    event.jetpt_rec = jet_rec.Pt()
    event.nGenAll = len(constituent_gen) if len(constituent_gen) > 0 else float('nan')
    event.nGenMatched = len(constituent_gen_matched) if len(constituent_gen) > 0 else float('nan')
    event.matchingEffi = float(len(constituent_gen_matched))/float(len(constituent_gen)) if len(constituent_gen) > 0 else float('nan')
    event.passSel = True

    # Now we have separate lists for matched and unmatched particles for both gen and rec.
    # We can easily form the matched triplets (where all 3 constituents are matched)
    # by getting all triplets from these lists. This even results in a 1-to-1 match:
    # triplets_matched_gen[i] is matched to triplets_matched_rec[i].
    # It is more complicated with the unmatched. Only forming triplets from the
    # unmatched lists would not give us triplets with 1 or 2 matched and 2 or 1
    # unmatched constituents.
    # Thus we keep track of the matching status by inserting a label.
    # Then we form triplets for only the matched list to get the matched triplets.
    # Then we form triplets for all (matched+unmatched) particles and select only those
    # where not all 3 particles are matched.
    # This gives us all possible triplets but we still have the matched in 1-to-1
    # matched lists!

    # tag the constituents to keep track of matched and unmatched constituents
    constituent_gen_matched_tagged = tag_objects(constituent_gen_matched, 'matched')
    constituent_rec_matched_tagged = tag_objects(constituent_rec_matched, 'matched')
    constituent_gen_unmatched_tagged = tag_objects(constituent_gen_unmatched, 'unmatched')
    constituent_rec_unmatched_tagged = tag_objects(constituent_rec_unmatched, 'unmatched')
    constituent_gen_all_tagged = constituent_gen_matched_tagged + constituent_gen_unmatched_tagged
    constituent_rec_all_tagged = constituent_rec_matched_tagged + constituent_rec_unmatched_tagged

    # First only form the triplets from the matched list
    if len(constituent_gen_matched) > 0:
        _, event.zeta_gen, event.weight_gen, _ = getTriplets_pp_TLorentz_label(jet_gen.Pt(), constituent_gen_matched_tagged, n=1, max_zeta=None, max_delta_zeta=None, delta_legs=None, shortest_side=None, log=False)
        _, event.zeta_rec, event.weight_rec, _ = getTriplets_pp_TLorentz_label(jet_rec.Pt(), constituent_rec_matched_tagged, n=1, max_zeta=None, max_delta_zeta=None, delta_legs=None, shortest_side=None, log=False)

    # Now form triplets from all constituents (matched+unmatched) but only use the
    # unmatched (where all three constituents are unmatched)
    if len(constituent_gen_all_tagged) > 0:
        # Get all triplets
        _, zeta_values, weights, triplet_labels = getTriplets_pp_TLorentz_label(jet_gen.Pt(), constituent_gen_all_tagged, n=1, max_zeta=None, max_delta_zeta=None, delta_legs=None, shortest_side=None, log=False)
        # Only select unmatched triplets
        labels = np.array(triplet_labels)
        unmatched_mask = labels == "unmatched"
        event.zeta_gen_unmatched = zeta_values[unmatched_mask]
        event.weight_gen_unmatched = weights[unmatched_mask]

    # Do the same for rec
    if len(constituent_rec_all_tagged) > 0:
        _, zeta_values, weights, triplet_labels = getTriplets_pp_TLorentz_label(jet_rec.Pt(), constituent_rec_all_tagged, n=1, max_zeta=None, max_delta_zeta=None, delta_legs=None, shortest_side=None, log=False)
        labels = np.array(triplet_labels)
        unmatched_mask = labels == "unmatched"
        event.zeta_rec_unmatched = zeta_values[unmatched_mask]
        event.weight_rec_unmatched = weights[unmatched_mask]


sequence.append( getConstituents )

def storeBWreweight( event, sample ):
    top_had = getHadronicTop(event)
    if top_had is not None:
        BW_factor_171p5 = BreitWignerReweight(width_old=1.3, width_new=1.3, peak_old=172.5, peak_new=171.5, mtop=top_had.M())
        BW_factor_173p5 = BreitWignerReweight(width_old=1.3, width_new=1.3, peak_old=172.5, peak_new=173.5, mtop=top_had.M())
        event.BW_reweight_171p5 = BW_factor_171p5
        event.BW_reweight_173p5 = BW_factor_173p5
        event.mtop = top_had.M()
    else:
        event.BW_reweight_171p5 = 1.0
        event.BW_reweight_173p5 = 1.0
        event.mtop = -1.0

sequence.append( storeBWreweight )

def calculateEventWeight( event, sample ):
    Nevents = sample.normalization
    xsection = sample.xSection
    lumi_weight = xsection/Nevents # first scale to 1pb
    event.event_weight_gen = float(event.Generator_weight)*lumi_weight*lumi_info["UL18"] # now multiply with gen weight and lumi factor
sequence.append(calculateEventWeight)
################################################################################
# Read variables

read_variables = [
    "nGenPart/I",
    "GenPart[pt/F,eta/F,phi/F,m/F,pdgId/I,mompdgId/I,grmompdgId/I]",
    "nGenJetAK8/I",
    "GenJetAK8[pt/F,eta/F,phi/F,mass/F]",
    "nGenJetAK8_cons/I",
    "Generator_weight/D",
    VectorTreeVariable.fromString( "GenJetAK8_cons[pt/F,eta/F,phi/F,mass/F,pdgId/I,jetIndex/I]", nMax=1000),
    "nPFJetAK8/I",
    "PFJetAK8[pt/F,eta/F,phi/F,mass/F]",
    "nPFJetAK8_cons/I",
    VectorTreeVariable.fromString( "PFJetAK8_cons[pt/F,eta/F,phi/F,mass/F,pdgId/I,jetIndex/I]", nMax=1000),
    "nElectron/I", "Electron[pt/F, eta/F, phi/F]",
    "nMuon/I", "Muon[pt/F, eta/F, phi/F]",
]

################################################################################
# Histograms
pt_windows = [(400,425),(425,450),(450,475),(475,500)]
histograms = {
    "N_const_rec": ROOT.TH1F("N_const_rec", "N_const_rec", 20, 0, 100),
    "N_const_nocut_rec": ROOT.TH1F("N_const_nocut_rec", "N_const_nocut_rec", 20, 0, 100),
    "N_const_onlypt_rec": ROOT.TH1F("N_const_onlypt_rec", "N_const_onlypt_rec", 20, 0, 100),
    "N_const_gen": ROOT.TH1F("N_const_gen", "N_const_gen", 20, 0, 100),
    "N_const_nocut_gen": ROOT.TH1F("N_const_nocut_gen", "N_const_nocut_gen", 20, 0, 100),
    "N_const_onlypt_gen": ROOT.TH1F("N_const_onlypt_gen", "N_const_onlypt_gen", 20, 0, 100),
    "zeta_ptscaled_rec": ROOT.TH1F("zeta_ptscaled_rec", "zeta_ptscaled_rec", 40, 0, 7),
    "particle_matching_effi": ROOT.TH1F("particle_matching_effi", "particle_matching_effi", 20, 0.0, 1.0),
    "triplet_matching_effi_gen": ROOT.TH1F("triplet_matching_effi_gen", "triplet_matching_effi_gen", 20, 0.0, 1.0),
    "triplet_matching_effi_rec": ROOT.TH1F("triplet_matching_effi_rec", "triplet_matching_effi_rec", 20, 0.0, 1.0),
}
for (pt_low, pt_high) in pt_windows:
    ptstring = "%ito%i"%(pt_low,pt_high)
    histograms["zeta_"+ptstring+"_rec"] = ROOT.TH1F("zeta_"+ptstring+"_rec", "zeta_"+ptstring+"_rec", 40, 0, 1)
    histograms["zeta_"+ptstring+"_mt1715_rec"] = ROOT.TH1F("zeta_"+ptstring+"_mt1715_rec", "zeta_"+ptstring+"_mt1715_rec", 40, 0, 1)
    histograms["zeta_"+ptstring+"_mt1735_rec"] = ROOT.TH1F("zeta_"+ptstring+"_mt1735_rec", "zeta_"+ptstring+"_mt1735_rec", 40, 0, 1)

for sample in mc:
    hist = histograms.copy()
    r = sample.treeReader( variables = read_variables, sequence = sequence, selectionString = cutInterpreter.cutString(args.selection_rec))
    r.start()
    new_tree = ROOT.TTree("Events", "Events")

    ievent = array.array('i', [-1])
    zeta_weight_rec = array.array('f', [0.])
    zeta_weight_gen = array.array('f', [0.])
    zeta_rec = array.array('f', [0.])
    zeta_gen = array.array('f', [0.])
    jetpt_rec = array.array('f', [0.])
    jetpt_gen = array.array('f', [0.])
    has_gen_info = array.array('i', [-1])
    has_rec_info = array.array('i', [-1])
    pass_triplet_top_gen = array.array('i', [-1])
    pass_triplet_top_rec = array.array('i', [-1])
    pass_triplet_W_gen = array.array('i', [-1])
    pass_triplet_W_rec = array.array('i', [-1])
    BW_reweight_171p5 = array.array('f', [0.])
    BW_reweight_173p5 = array.array('f', [0.])
    mtop = array.array('f', [0.])
    event_weight_gen = array.array('f', [0.])
    event_weight_rec = array.array('f', [0.])

    # define branches
    new_tree.Branch("ievent", ievent, "ievent/I")
    new_tree.Branch("zeta_weight_rec", zeta_weight_rec, "zeta_weight_rec/F")
    new_tree.Branch("zeta_weight_gen", zeta_weight_gen, "zeta_weight_gen/F")
    new_tree.Branch("zeta_rec", zeta_rec, "zeta_rec/F")
    new_tree.Branch("zeta_gen", zeta_gen, "zeta_gen/F")
    new_tree.Branch("jetpt_rec", jetpt_rec, "jetpt_rec/F")
    new_tree.Branch("jetpt_gen", jetpt_gen, "jetpt_gen/F")
    new_tree.Branch("has_gen_info", has_gen_info, "has_gen_info/I")
    new_tree.Branch("has_rec_info", has_rec_info, "has_rec_info/I")
    new_tree.Branch("pass_triplet_top_rec", pass_triplet_top_rec, "pass_triplet_top_rec/I")
    new_tree.Branch("pass_triplet_top_gen", pass_triplet_top_gen, "pass_triplet_top_gen/I")
    new_tree.Branch("pass_triplet_W_rec", pass_triplet_W_rec, "pass_triplet_W_rec/I")
    new_tree.Branch("pass_triplet_W_gen", pass_triplet_W_gen, "pass_triplet_W_gen/I")
    new_tree.Branch("BW_reweight_171p5", BW_reweight_171p5, "BW_reweight_171p5/F")
    new_tree.Branch("BW_reweight_173p5", BW_reweight_173p5, "BW_reweight_173p5/F")
    new_tree.Branch("mtop", mtop, "mtop/F")
    new_tree.Branch("event_weight_gen", event_weight_gen, "event_weight_gen/F")
    new_tree.Branch("event_weight_rec", event_weight_rec, "event_weight_rec/F")

    event_counter = 0
    while r.run():
        event = r.event
        event_counter += 1
        if event.passSel:
            ################################################################
            # Fill some event level hists
            if args.fillHists:
                histograms["N_const_gen"].Fill(event.nGenParts)
                histograms["N_const_nocut_gen"].Fill(event.nGenParts_noCut)
                histograms["N_const_onlypt_gen"].Fill(event.nGenParts_onlyPT)
                histograms["N_const_rec"].Fill(event.nPFParts)
                histograms["N_const_nocut_rec"].Fill(event.nPFParts_noCut)
                histograms["N_const_onlypt_rec"].Fill(event.nPFParts_onlyPT)
                histograms["particle_matching_effi"].Fill(event.matchingEffi)
                N_matched = float(len(event.zeta_gen))
                N_unmatched_gen = float(len(event.zeta_gen_unmatched))
                N_unmatched_rec = float(len(event.zeta_rec_unmatched))
                triplet_effi_gen = N_matched/(N_matched+N_unmatched_gen) if (N_matched+N_unmatched_gen)>0 else 0
                triplet_effi_rec = N_matched/(N_matched+N_unmatched_rec) if (N_matched+N_unmatched_rec)>0 else 0
                histograms["triplet_matching_effi_gen"].Fill( triplet_effi_gen )
                histograms["triplet_matching_effi_rec"].Fill( triplet_effi_rec )

            ################################################################
            # Fill new tree with matched triplets
            for i in range(len(event.zeta_gen)):
                ievent[0] = event_counter
                has_rec_info[0] = 1
                has_gen_info[0] = 1
                zeta_weight_gen[0] = event.weight_gen[i]
                zeta_weight_rec[0] = event.weight_rec[i]
                zeta_gen[0] = event.zeta_gen[i][0]
                zeta_rec[0] = event.zeta_rec[i][0]
                jetpt_gen[0] = event.jetpt_gen
                jetpt_rec[0] = event.jetpt_rec
                pass_triplet_top_gen[0] = 1 if passTripletSelection(event.zeta_gen[i], event.jetpt_gen, sel="top") else 0
                pass_triplet_top_rec[0] = 1 if passTripletSelection(event.zeta_rec[i], event.jetpt_rec, sel="top") else 0
                pass_triplet_W_gen[0] = 1 if passTripletSelection(event.zeta_gen[i], event.jetpt_gen, sel="W") else 0
                pass_triplet_W_rec[0] = 1 if passTripletSelection(event.zeta_rec[i], event.jetpt_rec, sel="W") else 0
                BW_reweight_171p5[0] = event.BW_reweight_171p5
                BW_reweight_173p5[0] = event.BW_reweight_173p5
                mtop[0] = event.mtop
                event_weight_gen[0] = event.event_weight_gen
                event_weight_rec[0] = 1.0
                new_tree.Fill()
                # Fill hists here
                if args.fillHists:
                    if pass_triplet_top_rec[0] == 1:
                        ptfactor_rec = event.jetpt_rec**2 / (172.5**2)
                        histograms["zeta_ptscaled_rec"].Fill(event.zeta_rec[i][0]*ptfactor_rec, event.weight_rec[i])
                        for (pt_low, pt_high) in pt_windows:
                            if event.jetpt_rec > pt_low and event.jetpt_rec < pt_high:
                                ptstring = "%ito%i"%(pt_low,pt_high)
                                histograms["zeta_"+ptstring+"_rec"].Fill(event.zeta_rec[i][0], event.weight_rec[i])
                                histograms["zeta_"+ptstring+"_mt1715_rec"].Fill(event.zeta_rec[i][0], event.weight_rec[i]*event.BW_reweight_171p5)
                                histograms["zeta_"+ptstring+"_mt1735_rec"].Fill(event.zeta_rec[i][0], event.weight_rec[i]*event.BW_reweight_173p5)




            ################################################################
            # Fill new tree with unmatched rec triplets
            for i in range(len(event.zeta_rec_unmatched)):
                ievent[0] = event_counter
                has_gen_info[0] = 0
                has_rec_info[0] = 1
                zeta_weight_gen[0] = -1
                zeta_weight_rec[0] = event.weight_rec_unmatched[i]
                zeta_gen[0] = -1
                zeta_rec[0] = event.zeta_rec_unmatched[i][0]
                jetpt_gen[0] = event.jetpt_gen
                jetpt_rec[0] = event.jetpt_rec
                pass_triplet_top_gen[0] = 0
                pass_triplet_top_rec[0] = 1 if  passTripletSelection(event.zeta_rec_unmatched[i], event.jetpt_rec, sel="top") else 0
                pass_triplet_W_gen[0] = 0
                pass_triplet_W_rec[0] = 1 if passTripletSelection(event.zeta_rec_unmatched[i], event.jetpt_rec, sel="W") else 0
                BW_reweight_171p5[0] = event.BW_reweight_171p5
                BW_reweight_173p5[0] = event.BW_reweight_173p5
                mtop[0] = event.mtop
                event_weight_gen[0] = event.event_weight_gen
                event_weight_rec[0] = 1.0
                new_tree.Fill()
                # Fill hists here
                if args.fillHists:
                    if pass_triplet_top_rec[0] == 1:
                        ptfactor_rec = event.jetpt_rec**2 / (172.5**2)
                        histograms["zeta_ptscaled_rec"].Fill(event.zeta_rec_unmatched[i][0]*ptfactor_rec, event.weight_rec_unmatched[i])
                        for (pt_low, pt_high) in pt_windows:
                            if event.jetpt_rec > pt_low and event.jetpt_rec < pt_high:
                                ptstring = "%ito%i"%(pt_low,pt_high)
                                histograms["zeta_"+ptstring+"_rec"].Fill(event.zeta_rec_unmatched[i][0], event.weight_rec_unmatched[i])
                                histograms["zeta_"+ptstring+"_mt1715_rec"].Fill(event.zeta_rec_unmatched[i][0], event.weight_rec_unmatched[i]*event.BW_reweight_171p5)
                                histograms["zeta_"+ptstring+"_mt1735_rec"].Fill(event.zeta_rec_unmatched[i][0], event.weight_rec_unmatched[i]*event.BW_reweight_173p5)
            ################################################################
            # Fill new tree with unmatched gen triplets
            for i in range(len(event.zeta_gen_unmatched)):
                ievent[0] = event_counter
                has_gen_info[0] = 1
                has_rec_info[0] = 0
                zeta_weight_gen[0] = event.weight_gen_unmatched[i]
                zeta_weight_rec[0] = -1
                zeta_gen[0] = event.zeta_gen_unmatched[i][0]
                zeta_rec[0] = -1
                jetpt_gen[0] = event.jetpt_gen
                jetpt_rec[0] = event.jetpt_rec
                pass_triplet_top_gen[0] = 1 if  passTripletSelection(event.zeta_gen_unmatched[i], event.jetpt_gen, sel="top") else 0
                pass_triplet_top_rec[0] = 0
                pass_triplet_W_gen[0] = 1 if passTripletSelection(event.zeta_gen_unmatched[i], event.jetpt_gen, sel="W") else 0
                pass_triplet_W_rec[0] = 0
                BW_reweight_171p5[0] = event.BW_reweight_171p5
                BW_reweight_173p5[0] = event.BW_reweight_173p5
                mtop[0] = event.mtop
                event_weight_gen[0] = event.event_weight_gen
                event_weight_rec[0] = 1.0
                new_tree.Fill()
    logger.info( "Done with sample "+sample.name+" and selectionString "+cutInterpreter.cutString(args.selection_rec) )

    outfilename = outdir+sample.name+".root"
    writeObjToFile(fname=outfilename, obj=new_tree, writename="Events", update=False)
    logger.info( "Saved TTree to file "+outfilename)
    if args.fillHists:
        outfilename_hists = outdir+sample.name+"_HISTS.root"
        for histname in histograms.keys():
            writeObjToFile(fname=outfilename_hists, obj=histograms[histname], writename=histname, update=True)
        logger.info( "Saved histograms to file "+outfilename_hists)
