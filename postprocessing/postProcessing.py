#!/usr/bin/env python
import ROOT
import sys
import os
import uuid

from RootTools.core.standard import *

from Analysis.Tools.helpers import checkRootFile, deepCheckWeight, dRCleaning, nonEmptyFile

import CorrelatorMtop.Tools.user as user
from CorrelatorMtop.Tools.helpers import deepCheckRootFile
from CorrelatorMtop.Tools.objectSelection import getMuons, getElectrons, muonSelector, eleSelector, getGoodMuons, getGoodElectrons, isBJet, getGenPartsAll, getJets

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser for cmgPostProcessing")
argParser.add_argument('--logLevel',    action='store',         nargs='?',  choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'],   default='INFO', help="Log level for logging" )
argParser.add_argument('--samples',     action='store',         nargs='*',  type=str, default=['TTToSemiLeptonic'],                 help="List of samples to be post-processed, given as CMG component name" )
argParser.add_argument('--nJobs',       action='store',         nargs='?',  type=int, default=1,                                    help="Number of jobs to run in total" )
argParser.add_argument('--job',         action='store',                     type=int, default=0,                                    help="Run only jobs i" )
argParser.add_argument('--targetDir',   action='store',         nargs='?',  type=str, default=user.postprocessing_output_directory, help="Name of the directory the post-processed files will be saved" )
argParser.add_argument('--processingEra', action='store',       nargs='?',  type=str, default='v0',                                 help="Name of the processing era" )
argParser.add_argument('--skim',        action='store',         nargs='?',  type=str, default='singlelep-boosted',                   help="Skim conditions to be applied for post-processing" )
argParser.add_argument('--year',        action='store',                     type=str,                                               help="Which year?" )
argParser.add_argument('--overwrite',   action='store_true',                                                                        help="Overwrite existing output files, bool flag set to True  if used" )
argParser.add_argument('--small',       action='store_true',                                                                        help="Run the file on a small sample (for test purpose), bool flag set to True if used" )
argParser.add_argument('--triggerSelection',            action='store_true',                                                        help="Apply trigger selection?" )
argParser.add_argument('--eventsPerJob',action='store',         nargs='?',  type=int, default=30000000,                             help="Maximum number of events per job (Approximate!)." )
args = argParser.parse_args()


if args.year not in [ 'UL2016', 'UL2016_preVFP', 'UL2017', 'UL2018' ]:
    raise Exception("Year %s not known"%year)
yearint = 2018
if args.year in ['UL2016', 'UL2016_preVFP']:
    yearint = 2016
elif "UL2017" == args.year:
    yearint = 2017
elif "UL2018" == args.year:
    yearint = 2018
################################################################################
# Logging
import CorrelatorMtop.Tools.logger as _logger
logFile = '/tmp/%s_%s_%s_njob%s.txt'%(args.skim, '_'.join(args.samples), os.environ['USER'], str(0 if args.nJobs==1 else args.job))
logger  = _logger.get_logger(args.logLevel, logFile = logFile)

# RootTools logger
import RootTools.core.logger as _logger_rt
logger_rt = _logger_rt.get_logger(args.logLevel, logFile = logFile )
################################################################################
# Set up skim
isSingleLep     = args.skim.lower().count('singlelep') > 0
isSmall         = args.skim.lower().count('small') > 0
isBoosted       = args.skim.lower().count('boosted') > 0

# Skim conditions
skimConds = []
if isSingleLep:
    skimConds.append( "Sum$(Electron_pt>30&&abs(Electron_eta)<2.5) + Sum$(Muon_pt>30&&abs(Muon_eta)<2.5)>=1" )
if isBoosted:
    skimConds.append( "Sum$(PFJetAK8_pt>300)>=1" )

################################################################################
#Samples: Load samples
maxN = 1 if args.small else None
if args.small:
    args.job = 0
    args.nJobs = 10000 # set high to just run over 1 input file

if args.year == "UL2016":
    allSamples = []
elif args.year == "UL2016_preVFP":
    allSamples = []
elif args.year == "UL2017":
    allSamples = []
elif args.year == "UL2018":
    from samples_unprocessed_UL2018 import allSamples as samples_UL2018
    allSamples = samples_UL2018

samples = []
for selected in args.samples:
    for sample in allSamples:
        if selected == sample.name:
            samples.append(sample)

if len(samples)==0:
    logger.info( "No samples found. Was looking for %s. Exiting" % args.samples )
    sys.exit(-1)

isData = False not in [s.isData for s in samples]
isMC   =  True not in [s.isData for s in samples]

# Check that all samples which are concatenated have the same x-section.
assert isData or len(set([s.xSection for s in samples]))==1, "Not all samples have the same xSection: %s !"%(",".join([s.name for s in samples]))
assert isMC or len(samples)==1, "Don't concatenate data samples"

xSection = samples[0].xSection if isMC else None

################################################################################
# Samples: combine if more than one
if len(samples)>1:
    sample_name =  samples[0].name+"_comb"
    logger.info( "Combining samples %s to %s.", ",".join(s.name for s in samples), sample_name )
    sample      = Sample.combine(sample_name, samples, maxN = maxN)
    sampleForPU = Sample.combine(sample_name, samples, maxN = -1)
elif len(samples)==1:
    sample      = samples[0]
    sampleForPU = samples[0]
else:
    raise ValueError( "Need at least one sample. Got %r",samples )

################################################################################
# small option
args.skim = args.skim + '_small' if args.small else args.skim

################################################################################
# LHE cut (e.g. on mtt < 700 for inclusive tt sample)
# TO BE IMPLEMENTED

################################################################################
# final output directory
storage_directory = os.path.join( args.targetDir, args.processingEra, args.year, args.skim, sample.name )
try:    #Avoid trouble with race conditions in multithreading
    os.makedirs(storage_directory)
    logger.info( "Created output directory %s.", storage_directory )
except:
    pass

################################################################################
# sort the list of files?
len_orig = len(sample.files)
sample = sample.split( n=args.nJobs, nSub=args.job)
logger.info( "fileBasedSplitting: Run over %i/%i files for job %i/%i."%(len(sample.files), len_orig, args.job, args.nJobs))
logger.debug("fileBasedSplitting: Files to be run over:\n%s", "\n".join(sample.files) )

################################################################################
# tmp_output_directory
tmp_output_directory  = os.path.join( user.postprocessing_tmp_directory, "%s_%s_%s_%s_%s"%(args.processingEra, args.year, args.skim, sample.name, str(uuid.uuid3(uuid.NAMESPACE_OID, sample.name))))
if os.path.exists(tmp_output_directory) and args.overwrite:
    if args.nJobs > 1:
        logger.warning( "NOT removing directory %s because nJobs = %i", tmp_output_directory, args.nJobs )
    else:
        logger.info( "Output directory %s exists. Deleting.", tmp_output_directory )
        shutil.rmtree(tmp_output_directory)

try:    #Avoid trouble with race conditions in multithreading
    os.makedirs(tmp_output_directory)
    logger.info( "Created output directory %s.", tmp_output_directory )
except:
    pass

# Anticipate & check output file
filename, ext = os.path.splitext( os.path.join(tmp_output_directory, sample.name + '.root') )
outfilename   = filename+ext

if not args.overwrite:
    if os.path.isfile(outfilename):
        logger.info( "Output file %s found.", outfilename)
        if checkRootFile( outfilename, checkForObjects=["Events"] ) and deepCheckRootFile( outfilename ) and deepCheckWeight( outfilename ):
            logger.info( "File already processed. Source: File check ok! Skipping." ) # Everything is fine, no overwriting
            sys.exit(0)
        else:
            logger.info( "File corrupt. Removing file from target." )
            os.remove( outfilename )
            logger.info( "Reprocessing." )
    else:
        logger.info( "Sample not processed yet." )
        logger.info( "Processing." )
else:
    logger.info( "Overwriting.")

# relocate original
sample.copy_files( os.path.join(tmp_output_directory, "input") )

################################################################################
# trigger selection
if args.triggerSelection and isSingleLep:
    electriggers = "HLT_Ele115_CaloIdVT_GsfTrkIdT"
    if args.year in ["UL2016","UL2016_preVFP"]:
        electriggers += "||HLT_Ele27_WPTight_Gsf||HLT_Photon175"
    elif args.year in ["UL2017"]:
        electriggers += "||HLT_Ele35_WPTight_Gsf||HLT_Photon200"
    elif args.year in ["UL2018"]:
        electriggers += "||HLT_Ele32_WPTight_Gsf||HLT_Photon200"

    muontriggers = "HLT_Mu50||HLT_TkMu50"
    triggerstring = "("+electriggers+"||"+muontriggers+")"

    logger.info("Sample will have the following trigger skim: %s"%triggerstring)
    skimConds.append( triggerstring )

# turn on all branches to be flexible for filter cut in skimCond etc.
sample.chain.SetBranchStatus("*",1)

# this is the global selectionString
selectionString = '&&'.join(skimConds)

################################################################################
# branches to be kept for data and MC
branchKeepStrings_DATAMC = [\
    "irun", "ilumi", "ievt", "nprim", "PV_npvsGood",
    "PuppiMET_*",
    "nPFJetAK4", "PFJetAK4_*",
    "nPFJetAK8", "PFJetAK8_*",
    "nElectron", "Electron_*",
    "nMuon", "Muon_*",
    "prefiringweight*",
]


#branches to be kept for MC samples only
branchKeepStrings_MC = [ "Generator_*", "GenPart_*", "nGenPart", "GENMET_*", "nGenJetAK4", "GenJetAK4_*", "nGenJetAK8", "GenJetAK8_*"]

#branches to be kept for data only
branchKeepStrings_DATA = [ ]

targetLumi = 1000 #pb-1 Which lumi to normalize to
if isData:
    lumiScaleFactor=None
    branchKeepStrings = branchKeepStrings_DATAMC + branchKeepStrings_DATA
    from FWCore.PythonUtilities.LumiList import LumiList
    # Apply golden JSON
    lumiList = LumiList(os.path.expandvars(sample.json))
    logger.info( "Loaded json %s", sample.json )
else:
    lumiScaleFactor = xSection*targetLumi/float(sample.normalization) if xSection is not None else None
    branchKeepStrings = branchKeepStrings_DATAMC + branchKeepStrings_MC

# jet variables
jetVars = [
    'pt/F', 'eta/F', 'phi/F', 'mass/F',
    'jetID/I', 'btag_DeepFlav/F',
]


# jes uncertainties
# jesUncertainties = [
#     "Total",
#     "AbsoluteMPFBias",
#     "AbsoluteScale",
#     "AbsoluteStat",
#     "RelativeBal",
#     "RelativeFSR",
#     "RelativeJEREC1",
#     "RelativeJEREC2",
#     "RelativeJERHF",
#     "RelativePtBB",
#     "RelativePtEC1",
#     "RelativePtEC2",
#     "RelativePtHF",
#     "RelativeSample",
#     "RelativeStatEC",
#     "RelativeStatFSR",
#     "RelativeStatHF",
#     "PileUpDataMC",
#     "PileUpPtBB",
#     "PileUpPtEC1",
#     "PileUpPtEC2",
#     "PileUpPtHF",
#     "PileUpPtRef",
#     "FlavorQCD",
#     "Fragmentation",
#     "SinglePionECAL",
#     "SinglePionHCAL",
#     "TimePtEta",
# ]

# if isMC:
#     jesVariations = ["pt_jes%s%s"%(var, upOrDown) for var in jesUncertainties for upOrDown in ["Up","Down"]]
#     jetVars     += ["%s/F"%var for var in jesVariations]
#     jetVars     += ['pt_jerUp/F', 'pt_jerDown/F', 'corr_JER/F', 'corr_JEC/F']
# else:
#     jesVariations = []

jetVarNames     = [x.split('/')[0] for x in jetVars]

# lepton variables
genLepVars      = ['pt/F', 'phi/F', 'eta/F', 'pdgId/I', 'genPartIdxMother/I', 'status/I', 'statusFlags/I']
genLepVarNames  = [x.split('/')[0] for x in genLepVars]
lepVars         = [
    'pt/F','eta/F','phi/F','pdgId/I', 'charge/I','deltaEtaSC/F',
    'mediumId/I', 'eleIndex/I','muIndex/I', 'jetIdx/I',
]
lepVarNames     = [x.split('/')[0] for x in lepVars]

# add event variables
read_variables = [
    TreeVariable.fromString('PuppiMET_pt/F'),
    TreeVariable.fromString('PuppiMET_phi/F'),
    TreeVariable.fromString('irun/I'),
    TreeVariable.fromString('ilumi/I'),
    TreeVariable.fromString('ievt/i'),
    TreeVariable.fromString('nprim/I'),
    TreeVariable.fromString('PV_npvsGood/I'),
    TreeVariable.fromString('prefiringweight/F'),
    TreeVariable.fromString('prefiringweightup/F'),
    TreeVariable.fromString('prefiringweightdown/F'),
    TreeVariable.fromString('nElectron/I'),
    VectorTreeVariable.fromString('Electron[pt/F,eta/F,phi/F,mvaid_Fallv2WP80_noIso/O]'),
    TreeVariable.fromString('nMuon/I'),
    VectorTreeVariable.fromString('Muon[pt/F,eta/F,phi/F,TightID/O]'),
    TreeVariable.fromString('nPFJetAK4/I'),
    VectorTreeVariable.fromString('PFJetAK4[%s]'% ( ','.join(jetVars) ) ),
]
# PF AK8 jet stuff
read_variables.append( TreeVariable.fromString('nPFJetAK8/I') )
read_variables.append( VectorTreeVariable.fromString('PFJetAK8[pt/F,eta/F,phi/F,mass/I]' ) )
read_variables.append( TreeVariable.fromString('nPFJetAK8_cons/I') )
read_variables.append( VectorTreeVariable.fromString("PFJetAK8_cons[pt/F,eta/F,phi/F,mass/F,pdgId/I,jetIndex/I]", nMax=1000)) # default nMax is 100, which would lead to corrupt values in this case


if isMC:
    read_variables.append( TreeVariable.fromString('GENMET_pt/F') )
    read_variables.append( TreeVariable.fromString('GENMET_phi/F') )
    read_variables.append( TreeVariable.fromString('nGenPart/I') )
    read_variables.append( VectorTreeVariable.fromString('GenPart[pt/F,m/F,phi/F,eta/F,pdgId/I,mompdgId/I,grmompdgId/I,status/I]', nMax=200 )) # default nMax is 100, which would lead to corrupt values in this case
    read_variables.append( TreeVariable.fromString('Generator_weight/F') )
    read_variables.append( TreeVariable.fromString('nGenJetAK4/I') )
    read_variables.append( VectorTreeVariable.fromString('GenJetAK4[pt/F,eta/F,phi/F,partonflav/I,hadronflav/I]' ) )
    # LHE weights
    read_variables.extend( ["nLHEScaleWeights/I" ] )
    read_variables.extend( ["nLHEPDFWeights/I" ] )
    read_variables.extend( ["nLHEPSWeights/I" ] )
    # Gen AK8 jet stuff
    read_variables.append( TreeVariable.fromString('nGenJetAK8/I') )
    read_variables.append( VectorTreeVariable.fromString('GenJetAK8[pt/F,eta/F,phi/F,mass/I]' ) )
    read_variables.append( TreeVariable.fromString('nGenJetAK8_cons/I') )
    read_variables.append( VectorTreeVariable.fromString("GenJetAK8_cons[pt/F,eta/F,phi/F,mass/F,pdgId/I,jetIndex/I]", nMax=1000)) # default nMax is 100, which would lead to corrupt values in this case


new_variables = [ 'weight/F', 'year/I', 'preVFP/O']

if isMC:
    new_variables.extend(['reweightPU/F','reweightPUUp/F','reweightPUDown/F', 'reweightL1Prefire/F', 'reweightL1PrefireUp/F', 'reweightL1PrefireDown/F'])

new_variables += [
    'nlep/I',
    'JetGood[%s]'% ( ','.join(jetVars+['index/I']) + ',genPt/F' ),
    'met_pt/F', 'met_phi/F',
]

if sample.isData: new_variables.extend( ['jsonPassed/I','isData/I'] )
new_variables.extend( ['nBTag/I', 'm3/F', 'minDLmass/F'] )

new_variables.append( 'lep[%s]'% ( ','.join(lepVars) + ',ptCone/F' + ',ptConeGhent/F'+ ',jetBTag/F' + ',mvaTOP/F' + ',mvaTOPv2/F' + ',jetPtRatio/F' +',jetNDauCharged/I'+',jetRelIso/F') )

if isSingleLep:
    new_variables.extend( ['nGoodMuons/I', 'nGoodElectrons/I', 'nGoodLeptons/I' ] )
    new_variables.extend( ['l1_pt/F', 'l1_eta/F', 'l1_phi/F', 'l1_pdgId/I', 'l1_index/I', 'l1_eleIndex/I', 'l1_muIndex/I'] )

# generator weights
if isMC:
    new_variables.append( TreeVariable.fromString("nScale/I") )
    new_variables.append( TreeVariable.fromString("Scale[Weight/F]") )
    new_variables.append( TreeVariable.fromString("nPDF/I") )
    new_variables.append( VectorTreeVariable.fromString("PDF[Weight/F]", nMax=150) ) # There are more than 100 PDF weights
    new_variables.append( TreeVariable.fromString("nPS/I") )
    new_variables.append( TreeVariable.fromString("PS[Weight/F]") )

# add systematic variations for jet variables
# for var in ['jerUp', 'jer', 'jerDown', 'unclustEnUp', 'unclustEnDown']:
#     if not var.startswith('unclust'):
#         new_variables.extend( ['nJetGood_'+var+'/I', 'nBTag_'+var+'/I'] )
# for uncert in jesUncertainties:
#     for upOrDown in ["Up", "Down"]:
#         var = "jes"+uncert+upOrDown
#         if not var.startswith('unclust'):
#             new_variables.extend( ['nJetGood_'+var+'/I', 'nBTag_'+var+'/I'] )

################################################################################
### nanoAOD postprocessor
from importlib import import_module
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor   import PostProcessor
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel       import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop       import Module

## modules for nanoAOD postprocessor

logger.info("Preparing nanoAOD postprocessing")
logger.info("Will put files into directory %s", tmp_output_directory)

logger.info("Using JERs for MET significance")
from PhysicsTools.NanoAODTools.postprocessing.modules.jme.jetmetHelperRun2 import *
METBranchName = 'MET'

# check if files are available (e.g. if dpm is broken this should result in an error)
for f in sample.files:
    if not checkRootFile(f):
        raise IOError ("File %s not available"%f)

# remove empty files. this is necessary in 2018 because empty miniAOD files exist.
sample.files = [ f for f in sample.files if nonEmptyFile(f) ]
newFileList = []

runPeriod = None
if sample.isData:
    runString = sample.name
    runString = runString.replace("_preVFP", "")
    runString = runString.replace("ver1", "")
    runString = runString.replace("ver2", "")
    runString = runString.split('_')[1]
    assert str(yearint) in runString, "Could not obtain run period from sample name %s" % sample.name
    runPeriod = runString[-1]

logger.info("Starting nanoAOD postprocessing")
for f in sample.files:
    # JMECorrector = createJMECorrector(
    #     isMC        = (not sample.isData),
    #     dataYear    = args.year,
    #     runPeriod   = runPeriod,
    #     jesUncert   = ",".join(jesUncertainties),
    #     jetType     = "AK4PFchs",
    #     metBranchName = METBranchName,
    #     isFastSim   = False,
    #     applySmearing = False)

    # modules = [ JMECorrector() ]
    modules = [ ]

    # need a hash to avoid data loss
    file_hash = str(hash(f))
    p = PostProcessor(tmp_output_directory, [f], cut=selectionString, modules=modules, postfix="_for_%s_%s"%(sample.name, file_hash))
    p.run()
    newFileList += [tmp_output_directory + '/' + f.split('/')[-1].replace('.root', '_for_%s_%s.root'%(sample.name, file_hash))]
logger.info("Done. Replacing input files for further processing.")

sample.files = newFileList
sample.clear()

# Define a reader
logger.info( "Running with selectionString %s", selectionString )

reader = sample.treeReader( \
    variables = read_variables,
    selectionString = selectionString
)

eleSelector_ = eleSelector( "presel_boosted", year = yearint, ptCut=50.0 )
muSelector_  = muonSelector("presel_boosted", year = yearint, ptCut=50.0 )


################################################################################
# FILL EVENT INFO HERE
################################################################################
def filler(event):
    r = reader.event
    event.isData = sample.isData
    event.year   = yearint
    event.preVFP = False
    if args.year == "UL2016_preVFP":
        event.preVFP = True

    # Gen weight and lumi weight
    logger.debug("Lumi weight")
    if isMC:
        if hasattr(r, "Generator_weight"):
            event.weight = lumiScaleFactor*r.Generator_weight if lumiScaleFactor is not None else 1
        else:
            event.weight = lumiScaleFactor if lumiScaleFactor is not None else 1
    elif sample.isData:
        event.weight = 1
    else:
        raise NotImplementedError( "isMC %r isData %r " % (isMC, isData) )

    # lumi lists and vetos
    logger.debug("Lumi lists")
    if sample.isData:
        event.jsonPassed  = lumiList.contains(r.run, r.ilumi)
        # apply JSON to data via event weight
        if not event.jsonPassed: event.weight=0
        # store decision to use after filler has been executed
        event.jsonPassed_ = event.jsonPassed

    ################################################################################
    # PU reweighting
    # if isMC and hasattr(r, "nprim"):
    #     from Analysis.Tools.puWeightsUL			import getPUReweight
    #     event.reweightPU     = getPUReweight( r.nprim, year=options.year, weight="nominal")
    #     event.reweightPUDown = getPUReweight( r.nprim, year=options.year, weight="down" )
    #     event.reweightPUUp   = getPUReweight( r.nprim, year=options.year, weight="up" )

    ################################################################################
    # Store Scale and PDF weights in a format that is readable with HEPHY framework
    if isMC:
        print(r.nLHEScaleWeights, r.nLHEPDFWeights, r.nLHEPSWeights)
        print(type(r.nLHEScaleWeights), type(r.nLHEPDFWeights), type(r.nLHEPSWeights))

        logger.debug("Found %i Scale weights"%(r.nLHEScaleWeights))
        scale_weights = [reader.sample.chain.GetLeaf("LHEScaleWeights").GetValue(i_weight) for i_weight in range(r.nLHEScaleWeights)]
        for i,w in enumerate(scale_weights):
            event.Scale_Weight[i] = w
        event.nScale = r.nLHEScaleWeights

        logger.debug("Found %i PDF weights"%(r.nLHEPDFWeights))
        pdf_weights = [reader.sample.chain.GetLeaf("LHEPDFWeights").GetValue(i_weight) for i_weight in range(r.nLHEPDFWeights)]
        for i,w in enumerate(pdf_weights):
            event.PDF_Weight[i] = w
        event.nPDF = r.nLHEPDFWeights

        logger.debug("Found %i PS weights"%(r.nLHEPSWeights))
        ps_weights = [reader.sample.chain.GetLeaf("LHEPSWeights").GetValue(i_weight) for i_weight in range(r.nLHEPSWeights)]
        for i,w in enumerate(ps_weights):
            event.PS_Weight[i] = w
        event.nPS = r.nLHEPSWeights

    logger.debug("Prefire weights")
    event.reweightL1Prefire, event.reweightL1PrefireUp, event.reweightL1PrefireDown = r.prefiringweight, r.prefiringweightup, r.prefiringweightdown

    # get electrons and muons
    logger.debug("Get Leptons")
    electrons_pt50  = getGoodElectrons(r, collVars = ['pt','eta','phi','mvaid_Fallv2WP80_noIso'], ele_selector = eleSelector_)
    muons_pt50      = getGoodMuons    (r, collVars = ['pt','eta','phi','TightID'], mu_selector  = muSelector_ )

    # make list of leptons
    leptons = electrons_pt50+muons_pt50
    leptons.sort(key = lambda p:-p['pt'])

    # STORE MEASUREMENT JET IN EXTRA VARIABLE
    # 1. find leading lepton
    # 2. find leptonic jet
    # 3. hadronic jec = leading jet that is not leptonic

################################################################################
################################################################################
################################################################################

# Create a maker. Maker class will be compiled. This instance will be used as a parent in the loop
treeMaker_parent = TreeMaker(
    sequence  = [ filler ],
    variables = [ TreeVariable.fromString(x) if type(x)==type("") else x for x in new_variables ],
    treeName = "Events"
)

# Split input in ranges
eventRanges = reader.getEventRanges( maxNEvents = args.eventsPerJob, minJobs = 1 )

logger.info( "Splitting into %i ranges of %i events on average. FileBasedSplitting: %s. Job number %s",
        len(eventRanges),
        (eventRanges[-1][1] - eventRanges[0][0])/len(eventRanges),
        'Yes',
        args.job)

#Define all jobs
jobs = [(i, eventRanges[i]) for i in range(len(eventRanges))]

filename, ext = os.path.splitext( os.path.join(tmp_output_directory, sample.name + '.root') )

if len(eventRanges)>1:
    raise RuntimeError("Using fileBasedSplitting but have more than one event range!")

clonedEvents = 0
convertedEvents = 0
outputLumiList = {}
for ievtRange, eventRange in enumerate( eventRanges ):

    logger.info( "Processing range %i/%i from %i to %i which are %i events.",  ievtRange, len(eventRanges), eventRange[0], eventRange[1], eventRange[1]-eventRange[0] )

    _logger.   add_fileHandler( outfilename.replace('.root', '.log'), args.logLevel )
    _logger_rt.add_fileHandler( outfilename.replace('.root', '_rt.log'), args.logLevel )

    tmp_gdirectory = ROOT.gDirectory
    logger.info(f"Open output file: {outfilename}")
    outputfile = ROOT.TFile.Open(outfilename, 'recreate')
    tmp_gdirectory.cd()

    if args.small:
        logger.info("Running 'small'. Not more than 10000 events")
        nMaxEvents = eventRange[1]-eventRange[0]
        eventRange = ( eventRange[0], eventRange[0] +  min( [nMaxEvents, 10000] ) )

    # Set the reader to the event range
    reader.setEventRange( eventRange )

    clonedTree = reader.cloneTree( branchKeepStrings, newTreename = "Events", rootfile = outputfile )
    clonedEvents += clonedTree.GetEntries()

    # Clone the empty maker in order to avoid recompilation at every loop iteration
    maker = treeMaker_parent.cloneWithoutCompile( externalTree = clonedTree )

    maker.start()
    # Do the thing
    reader.start()
    reader.sample.chain.SetBranchStatus("*",1)
    while reader.run():

        maker.run()
        if sample.isData:
            if maker.event.jsonPassed_:
                if reader.event.irun not in outputLumiList.keys():
                    outputLumiList[reader.event.run] = set([reader.event.ilumi])
                else:
                    if reader.event.ilumi not in outputLumiList[reader.event.irun]:
                        outputLumiList[reader.event.irun].add(reader.event.ilumi)

    convertedEvents += maker.tree.GetEntries()
    maker.tree.Write()
    outputfile.Close()
    logger.info( "Written %s", outfilename)

    # Destroy the TTree
    maker.clear()
    sample.clear()


logger.info( "Converted %i events of %i, cloned %i",  convertedEvents, reader.nEvents , clonedEvents )

# Storing JSON file of processed events
if sample.isData and convertedEvents>0: # avoid json to be overwritten in cases where a root file was found already
    jsonFile = filename+'_%s.json'%(0 if args.nJobs==1 else args.job)
    LumiList( runsAndLumis = outputLumiList ).writeJSON(jsonFile)
    logger.info( "Written JSON file %s", jsonFile )

for f in sample.files:
    try:
        os.remove(f)
        logger.info("Removed nanoAOD file: %s", f)
    except OSError:
        logger.info("nanoAOD file %s seems to be not there", f)

logger.info("Copying log file to %s", storage_directory )
copyLog = subprocess.call(['cp', logFile, storage_directory] )
if copyLog:
    logger.info( "Copying log from %s to %s failed", logFile, storage_directory)
else:
    logger.info( "Successfully copied log file" )
    os.remove(logFile)
    logger.info( "Removed temporary log file" )

if checkRootFile( outfilename, checkForObjects=["Events"] ) and deepCheckWeight( outfilename ):
    logger.info( "Target: File check ok!" )
else:
    logger.info( "Corrupt rootfile! Removing file: %s"%outfilename )
    os.remove( outfilename )

for item in os.listdir(tmp_output_directory):
    s = os.path.join(tmp_output_directory, item)
    if not os.path.isdir(s):
        shutil.copy(s, storage_directory)
logger.info( "Done copying to storage directory %s", storage_directory)

# close all log files before deleting the tmp directory
for logger_ in [logger, logger_rt]:
    for handler in logger_.handlers:
        handler.close()
        logger_.removeHandler(handler)

if os.path.exists(tmp_output_directory):
    shutil.rmtree(tmp_output_directory)
    logger.info( "Cleaned tmp directory %s", tmp_output_directory )

# There is a double free corruption due to stupid ROOT memory management which leads to a non-zero exit code
# Thus the job is resubmitted on condor even if the output is ok
# Current idea is that the problem is with xrootd having a non-closed root file
sample.clear()
