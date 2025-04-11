# CorrelatorMtop

```
cmsrel CMSSW_14_0_0
cd CMSSW_14_0_0/src
cmsenv
git cms-init
git clone https://github.com/HephyAnalysisSW/CorrelatorMtop
./CorrelatorMtop/setup.sh
cd $CMSSW_BASE
curl -sLO https://gist.githubusercontent.com/dietrichliko/8aaeec87556d6dd2f60d8d1ad91b4762/raw/a34563dfa03e4db62bb9d7bf8e5bf0c1729595e3/install_correctionlib.sh
. ./install_correctionlib.sh
scram b -j10
```
