#include <iostream>
#include <fstream>
#include <TFile.h>
#include "TUnfoldBinningXML.h"
#include <TF2.h>

using namespace std;

Double_t* get_binning(int Nbins, Double_t min, Double_t max, bool log) {
  Double_t* bins = new Double_t[Nbins + 1];
  if(log){
    if(min <= 0 || max <= 0) throw std::invalid_argument("Logarithmic binning requires min and max to be > 0.");
    Double_t log_min = std::log10(min);
    Double_t log_max = std::log10(max);
    Double_t step = (log_max - log_min) / Nbins;
    for (int i = 0; i <= Nbins; ++i) {
        bins[i] = std::pow(10, log_min + i * step);
        cout << bins[i] << endl;
    }
  }
  else{
    Double_t step = (max - min) / Nbins;
    for (int i = 0; i <= Nbins; ++i) {
        bins[i] = min + i * step;
    }
  }
  return bins;
}

int main(int argc, char* argv[])
{
  cout << "create root file" << endl;

  // create root file to store binning schemes

  TString binning_root;
  TString binning_xml;

  binning_root = "Binning.root";
  binning_xml = "Binning.xml";

  TFile *binningSchemes=new TFile(binning_root,"recreate");


  cout << "set up binning" << endl;

  /******************* RECO BINNING ***********************************/
  int N_BINS_REC_ZETA = 10;
  Double_t* BINS_REC_ZETA = get_binning(N_BINS_REC_ZETA, 1.0, 7.0, false);

  int N_BINS_REC_WEIGHT = 10;
  Double_t* BINS_REC_WEIGHT = get_binning(N_BINS_REC_WEIGHT, 0.00000001, 0.0001, true);

  /******************* GEN BINNING ************************************/
  int N_BINS_GEN_ZETA = 7;
  Double_t* BINS_GEN_ZETA = get_binning(N_BINS_GEN_ZETA, 1.0, 7.0, false);

  int N_BINS_GEN_WEIGHT = 7;
  Double_t* BINS_GEN_WEIGHT = get_binning(N_BINS_GEN_WEIGHT, 0.00000001, 0.0001, true);



  // =======================================================================================================
  //
  // REC BINNING
  //
  TUnfoldBinning *binning_rec=new TUnfoldBinning("binning_rec");

  //
  // define measurement phase space distribution
  //
  TUnfoldBinning *measurement_rec = binning_rec->AddBinning("measurement_rec");
  measurement_rec->AddAxis("zeta",N_BINS_REC_ZETA, BINS_REC_ZETA,
                                false, // underflow bin
                                false // overflow bin
                                );
  measurement_rec->AddAxis("weight",N_BINS_REC_WEIGHT, BINS_REC_WEIGHT,
                                false, // no underflow bin
                                true // overflow bin
                                );

  // =======================================================================================================
  //
  // GEN BINNING
  //
  TUnfoldBinning *binning_gen=new TUnfoldBinning("binning_gen");

  //
  // define measurement phase space distribution
  //
  TUnfoldBinning *measurement_gen = binning_gen->AddBinning("measurement_gen");
  measurement_gen->AddAxis("zeta",N_BINS_GEN_ZETA, BINS_GEN_ZETA,
                                false, // underflow bin
                                false // overflow bin
                                );
  measurement_gen->AddAxis("weight",N_BINS_GEN_WEIGHT, BINS_GEN_WEIGHT,
                                false, // no underflow bin
                                true // overflow bin
                                );



  cout << "wirte binning scheme to root file" << endl;
  binning_rec->Write();
  binning_gen->Write();



  cout << "wirte binning scheme to readable txt file" << endl;
  // also write binning scheme as readable to a txt file
  std::ofstream out("Binning.txt");
  auto coutbuf = std::cout.rdbuf(out.rdbuf());
  cout << "BINNING REC" << endl;
  vector<TUnfoldBinning*> regions_rec = {measurement_rec};
  vector<TString> names_rec = {"measurement_rec"};
  for(unsigned int i=0; i<regions_rec.size(); i++){
    int start = regions_rec[i]->GetStartBin();
    int end = regions_rec[i]->GetEndBin() - 1;
    cout << "  -- " << names_rec[i] << " (bins "<< start << " - " << end << ")" << endl;
  }
  cout << "-------------------------------------------" << endl;
  cout << "BINNING GEN" << endl;
  vector<TUnfoldBinning*> regions_gen = {measurement_gen};
  vector<TString> names_gen = {"measurement_gen"};
  for(unsigned int i=0; i<regions_gen.size(); i++){
    int start = regions_gen[i]->GetStartBin();
    int end = regions_gen[i]->GetEndBin() - 1;
    cout << "  -- " << names_gen[i] << " (bins "<< start << " - " << end << ")" << endl;
  }
  std::cout.rdbuf(coutbuf);
  ////

  cout << "wirte binning scheme to xml file" << endl;
  ofstream xmlOut(binning_xml);
  TUnfoldBinningXML::ExportXML(*binning_rec,xmlOut,kTRUE,kFALSE);
  TUnfoldBinningXML::ExportXML(*binning_gen,xmlOut,kFALSE,kTRUE);
  TUnfoldBinningXML::WriteDTD();
  xmlOut.close();

  delete binningSchemes;

  cout << "finished" << endl;
  return 0;
}
