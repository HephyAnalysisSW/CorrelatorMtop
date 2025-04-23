#
eval `scram runtime -sh`
cd $CMSSW_BASE/src

# nanoAOD tools (for MET Significance, JEC/JER...)
git clone https://github.com/cms-nanoAOD/nanoAOD-tools.git PhysicsTools/NanoAODTools
cd $CMSSW_BASE/src

# RootTools (for plotting, sample handling, processing)
git clone https://github.com/HephyAnalysisSW/RootTools.git --branch python3
cd $CMSSW_BASE/src

# Shared samples (miniAOD/nanoAOD)
# git clone https://github.com/HephyAnalysisSW/Samples.git
# cd $CMSSW_BASE/src

# Shared analysis tools and data
git clone https://github.com/HephyAnalysisSW/Analysis.git --branch python3
cd $CMSSW_BASE/src

cd $CMSSW_BASE/src/CorrelatorMtop

#compile
eval `scram runtime -sh`
cd $CMSSW_BASE/src && scram b -j 8
