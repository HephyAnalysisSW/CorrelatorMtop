# CorrelatorMtop

Framework for ML unfolding and binned unfolding using TUnfold for a measurement of energy correlators.

## Setup
```
cmsrel CMSSW_10_6_28
cd CMSSW_10_6_28/src
cmsenv
git cms-init
git clone git@github.com:HephyAnalysisSW/CorrelatorMtop.git
./CorrelatorMtop/setup.sh
cd $CMSSW_BASE
curl -sLO https://gist.githubusercontent.com/dietrichliko/8aaeec87556d6dd2f60d8d1ad91b4762/raw/a34563dfa03e4db62bb9d7bf8e5bf0c1729595e3/install_correctionlib.sh
. ./install_correctionlib.sh
scram b -j10
```

Then, put your user name in `Tools/python/user.py`.

## CMSSW version

Tested with CMSSW10.
There might be issues in compiling TUnfold in newer versions.
Everything else should run.

## Usage
There are separate instructions for:
Producing ntuples in `plots/plotsDennis/ntuple_maker/`, ML unfolding in `plots/plotsDennis/MLUnfold/`, and binned unfolding in `plots/plotsDennis/TUnfold/`.
