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
from CorrelatorMtop.Tools.user            import plot_directory
from CorrelatorMtop.Tools.cutInterpreter            import cutInterpreter
from CorrelatorMtop.Tools.energyCorrelators         import getTriplets_pp_TLorentz
from CorrelatorMtop.Tools.helpers              import deltaPhi, deltaR, deltaRTLorentz, writeObjToFile

import Analysis.Tools.syncer
import numpy as np

################################################################################
# Arguments
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',       action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--plot_directory', action='store', default='CorrelatorMtop_v1')
argParser.add_argument('--small',          action='store_true', help='Run only on a small subset of the data?', )
argParser.add_argument('--selection_rec',  action='store', default='PFboosted-PFleppt')
argParser.add_argument('--selection_gen',  action='store', default='GENboosted')
argParser.add_argument('--era',            action='store', type=str, default="UL2018")
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

mc = [TTToSemiLeptonic_UL2018]
# lumi_scale = 60

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

def getChargedParticlesFromJet(event, jetidx, genrec):
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
            charge = 1 if event.PFJetAK8_cons_pdgId[iPart]>0 else -1
            particles.append( (particle, charge) )
    return particles

################################################################################
# Define sequences
sequence       = []


def getConstituents( event, sample ):
    # Define all variables already here
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

    # Get leading lepton and leading jet
    # Check that they separated and get jet constituent
    lep_gen = getLeadingLepton(event, "gen")
    jet_gen, jet_gen_id = getLeadingJet(event, "gen")
    if lep_gen is None or jet_gen is None:
        return
    if deltaRTLorentz(lep_gen, jet_gen) < 0.8:
        return
    constituent_gen = getChargedParticlesFromJet(event, jet_gen_id, "gen")
    event.nGenParts = len(constituent_gen)

    # Do the same for PF jets
    lep_rec = getLeadingLepton(event, "rec")
    jet_rec, jet_rec_id = getLeadingJet(event, "rec")
    if lep_rec is None or jet_rec is None:
        return
    if deltaRTLorentz(lep_rec, jet_rec) < 0.8:
        return
    constituent_rec = getChargedParticlesFromJet(event, jet_gen_id, "gen")
    event.nPFParts = len(constituent_rec)

    # Match constituents
    maxDR_part = 0.05
    genMatches = {}
    alreadyMatched = []
    for i, (genPart, genCharge) in enumerate(constituent_gen):
        matches = []
        for j, (pfPart, pfCharge) in enumerate(constituent_rec):
            if j in alreadyMatched:
                continue
            if genCharge == pfCharge and genPart.DeltaR(pfPart) < maxDR_part:
                matches.append(j)
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
                    gmatchIDX = idx
        genMatches[i] = matchIDX
        alreadyMatched.append(matchIDX)

    constituent_gen_matched = []
    constituent_rec_matched = []
    for i, (genPart, genCharge) in enumerate(constituent_gen):
        if genMatches[i] is not None:
            constituent_gen_matched.append(constituent_gen[i][0])
            constituent_rec_matched.append(constituent_rec[genMatches[i]][0])

    event.nGenAll = len(constituent_gen) if len(constituent_gen) > 0 else float('nan')
    event.nGenMatched = len(constituent_gen_matched) if len(constituent_gen) > 0 else float('nan')
    event.matchingEffi = float(len(constituent_gen_matched))/float(len(constituent_gen)) if len(constituent_gen) > 0 else float('nan')
    if len(constituent_gen_matched) > 0:
        _, event.zeta_gen, _, _, event.weight_gen = getTriplets_pp_TLorentz(jet_gen.Pt(), constituent_gen_matched, n=1, max_zeta=None, max_delta_zeta=None, delta_legs=None, shortest_side=None, log=False)
        _, event.zeta_rec, _, _, event.weight_rec = getTriplets_pp_TLorentz(jet_rec.Pt(), constituent_rec_matched, n=1, max_zeta=None, max_delta_zeta=None, delta_legs=None, shortest_side=None, log=False)
    event.passSel = True

sequence.append( getConstituents )


################################################################################
# Read variables

read_variables = [
    "nGenPart/I",
    "GenPart[pt/F,eta/F,phi/F,m/F,pdgId/I,mompdgId/I,grmompdgId/I]",
    "nGenJetAK8/I",
    "GenJetAK8[pt/F,eta/F,phi/F,mass/F]",
    "nGenJetAK8_cons/I",
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
histograms = {
    "Weight_gen": ROOT.TH1F("Weight_gen", "Weight_gen", 100, 0, 0.04),
    "Weight_rec": ROOT.TH1F("Weight_rec", "Weight_rec", 100, 0, 0.04),
    "WeightZoom_gen": ROOT.TH1F("WeightZoom_gen", "WeightZoom_gen", 100, 0, 0.002),
    "WeightZoom_rec": ROOT.TH1F("WeightZoom_rec", "WeightZoom_rec", 100, 0, 0.002),
    "Weight_matrix": ROOT.TH2F("Weight_matrix", "Weight_matrix", 100, 0, 0.04, 100, 0, 0.04),
    "Zeta_gen": ROOT.TH1F("Zeta_gen", "Zeta_gen", 100, 0, 3.0),
    "Zeta_rec": ROOT.TH1F("Zeta_rec", "Zeta_rec", 100, 0, 3.0),
    "ZetaNoWeight_gen": ROOT.TH1F("ZetaNoWeight_gen", "ZetaNoWeight_gen", 100, 0, 3.0),
    "ZetaNoWeight_rec": ROOT.TH1F("ZetaNoWeight_rec", "ZetaNoWeight_rec", 100, 0, 3.0),
    "ZetaNoWeight_matrix": ROOT.TH2F("ZetaNoWeight_matrix", "ZetaNoWeight_matrix", 100, 0, 3.0, 100, 0, 3.0),
    "MatchingEfficiency": ROOT.TH1F("MatchingEfficiency", "MatchingEfficiency", 50, 0, 1.0),
}

for sample in mc:
    hist = histograms.copy()
    r = sample.treeReader( variables = read_variables, sequence = sequence, selectionString = cutInterpreter.cutString(args.selection_rec))
    r.start()
    new_tree = ROOT.TTree("Events", "Events")

    # Define variables for the branches
    # zeta_weight_rec = ROOT.std.vector('float')()
    # zeta_weight_gen = ROOT.std.vector('float')()
    # zeta_rec = ROOT.std.vector('float')()
    # zeta_gen = ROOT.std.vector('float')()
    zeta_weight_rec = array.array('f', [0.])
    zeta_weight_gen = array.array('f', [0.])
    zeta_rec = array.array('f', [0.])
    zeta_gen = array.array('f', [0.])

    # define branches
    # new_tree.Branch("zeta_weight_rec", zeta_weight_rec)
    # new_tree.Branch("zeta_weight_gen", zeta_weight_gen)
    # new_tree.Branch("zeta_rec", zeta_rec)
    # new_tree.Branch("zeta_gen", zeta_gen)
    new_tree.Branch("zeta_weight_rec", zeta_weight_rec, "zeta_weight_rec/F")
    new_tree.Branch("zeta_weight_gen", zeta_weight_gen, "zeta_weight_gen/F")
    new_tree.Branch("zeta_rec", zeta_rec, "zeta_rec/F")
    new_tree.Branch("zeta_gen", zeta_gen, "zeta_gen/F")


    while r.run():
        event = r.event
        if event.passSel:
            hist["MatchingEfficiency"].Fill(event.matchingEffi)
            for i in range(len(event.zeta_gen)):
                ################################################################
                # Fill new tree
                zeta_weight_gen[0] = event.weight_gen[i]
                zeta_weight_rec[0] = event.weight_rec[i]
                zeta_rec[0] = event.zeta_gen[i][0]
                zeta_gen[0] = event.zeta_rec[i][0]
                new_tree.Fill()
                ################################################################

                hist["Weight_gen"].Fill(event.weight_gen[i])
                hist["Weight_rec"].Fill(event.weight_rec[i])
                hist["WeightZoom_gen"].Fill(event.weight_gen[i])
                hist["WeightZoom_rec"].Fill(event.weight_rec[i])
                hist["Weight_matrix"].Fill(event.weight_gen[i], event.weight_rec[i])
                hist["ZetaNoWeight_gen"].Fill(event.zeta_gen[i][0])
                hist["ZetaNoWeight_rec"].Fill(event.zeta_rec[i][0])
                hist["ZetaNoWeight_matrix"].Fill(event.zeta_gen[i][0], event.zeta_rec[i][0])
                hist["Zeta_gen"].Fill(event.zeta_gen[i][0], event.weight_gen[i])
                hist["Zeta_rec"].Fill(event.zeta_rec[i][0], event.weight_rec[i])
    logger.info( "Done with sample "+sample.name+" and selectionString "+cutInterpreter.cutString(args.selection_rec) )

    outfilename = outdir+sample.name+".root"
    writeObjToFile(fname=outfilename, obj=new_tree, writename="Events", update=False)
    # outfile = ROOT.TFile(outfilename, "RECREATE")
    # outfile.cd()
    # for histname in hist.keys():
    #     hist[histname].Write(histname)
    # new_tree.Write()
    # outfile.Close()
    logger.info( "Saved histograms and new tree to file "+outfilename)
