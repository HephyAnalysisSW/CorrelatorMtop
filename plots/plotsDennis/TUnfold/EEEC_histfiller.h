#pragma once
#include <iostream>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TDOMParser.h>
#include <TXMLDocument.h>
#include "TUnfoldBinningXML.h"
#include <vector>
#include "SampleHistogram.h"

using namespace std;
void fill_hists(SampleHistogram& sample, TString BW_weight_name="none");
TFile *outputFile;
int Nentries_max = -1;

// binning schemes
const TUnfoldBinning *binning_gen;
const TUnfoldBinning *binning_rec;
const TUnfoldBinning *measurement_gen;
const TUnfoldBinning *measurement_rec;


// variables to store gen or rec info
Float_t zeta_gen, zeta_rec;
Float_t zeta_weight_gen, zeta_weight_rec;
Float_t jetpt_gen, jetpt_rec;
Float_t event_weight_gen, event_weight_rec;
Int_t i_event;
Float_t BW_factor;
Float_t mtop;
Int_t has_gen_info, has_rec_info;
Int_t pass_triplet_top_gen, pass_triplet_top_rec;
