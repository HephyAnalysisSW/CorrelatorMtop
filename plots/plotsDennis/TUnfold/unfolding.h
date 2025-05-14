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

class unfolding{
  public:
    unfolding(TH1F*, TH2F*, TUnfoldBinning*, TUnfoldBinning*);
    void set_offset_scale(double);
    void unfold();
    void set_fixed_tau(double);
    TH1F* get_unfolded();
    TH2F* get_unfolded_2D();

    enum class unfoldingMode {
        LScan,
        RhoScan,
        FixedTau
    };

    void set_unfolding_mode(unfoldingMode);


  private:
    bool unfolding_done;
    TH2F* h_response_matrix;
    TH1F* h_input;
    TUnfoldBinning* binning_rec;
    TUnfoldBinning* binning_gen;

    unfoldingMode uMode = unfoldingMode::LScan;
    TUnfold::EHistMap histMode;
    TUnfold::EConstraint constraintMode;
    TUnfold::ERegMode regMode;
    TUnfoldDensity::EDensityMode densityFlags;
    TUnfoldDensity::EScanTauMode scanMode;
    char *REGULARISATION_DISTRIBUTION;
    char *REGULARISATION_AXISSTEERING;
    char *SCAN_DISTRIBUTION;
    char *SCAN_AXISSTEERING;
    double scale_fb0 = 1.0;
    TSpline *rhoLogTau=0;
    TGraph *lcurve = 0;
    double lcurveX, lcurveY;
    double tau;
    double tau_fix = 0.0;
    TH1F* h_unfolded;
    TH2F* h_unfolded_2D;


};
