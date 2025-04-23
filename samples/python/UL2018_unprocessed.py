import copy, os, sys
from RootTools.core.Sample import Sample
import ROOT

# Logging
import logging
logger = logging.getLogger(__name__)

allSamples = []

TTToSemiLeptonic = Sample.fromDirectory(name="TTToSemiLeptonic", treeName="Events", isData=False, directory='/scratch-cbe/users/dennis.schwarz/MTopCorrelations/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8', normalization=2814182393.862684, xSection=831.762*(3*0.108)*(1-3*0.108)*2)
allSamples.append(TTToSemiLeptonic)
