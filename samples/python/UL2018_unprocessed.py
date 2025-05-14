import copy, os, sys
from RootTools.core.Sample import Sample
import ROOT

# Logging
import logging
logger = logging.getLogger(__name__)

allSamples = []

TTToSemiLeptonic = Sample.fromDirectory(name="TTToSemiLeptonic", treeName="Events", isData=False, directory='/scratch-cbe/users/dennis.schwarz/MTopCorrelations/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8', normalization=17230816913.4127, xSection=831.762*(3*0.108)*(1-3*0.108)*2)
allSamples.append(TTToSemiLeptonic)

# TTToSemiLeptonic_mtop171p5 = Sample.fromDirectory(name="TTToSemiLeptonic_mtop171p5", treeName="Events", isData=False, directory='/scratch-cbe/users/dennis.schwarz/MTopCorrelations/TTToSemiLeptonic_mtop171p5_TuneCP5_13TeV-powheg-pythia8', normalization=1.0, xSection=831.762*(3*0.108)*(1-3*0.108)*2)
# allSamples.append(TTToSemiLeptonic_mtop171p5)
#
# TTToSemiLeptonic_mtop173p5 = Sample.fromDirectory(name="TTToSemiLeptonic_mtop173p5", treeName="Events", isData=False, directory='/scratch-cbe/users/dennis.schwarz/MTopCorrelations/TTToSemiLeptonic_mtop173p5_TuneCP5_13TeV-powheg-pythia8', normalization=1.0, xSection=831.762*(3*0.108)*(1-3*0.108)*2)
# allSamples.append(TTToSemiLeptonic_mtop173p5)
