#pragma once
#include <iostream>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TDOMParser.h>
#include <TXMLDocument.h>
#include "TUnfoldBinningXML.h"
#include <vector>
#include "TUnfoldDensity.h"


using namespace std;

// binning schemes
TUnfoldBinning *binning_rec;
TUnfoldBinning *binning_gen;

TSpline *rhoLogTau=0;
TGraph *lcurve = 0;
