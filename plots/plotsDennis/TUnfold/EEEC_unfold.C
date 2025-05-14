#include "EEEC_unfold.h"
#include "unfolding.h"
#include <typeinfo>


int main(int argc, char* argv[]){
  unfolding::unfoldingMode umode =  unfolding::unfoldingMode::FixedTau;
  TString mode = "fixedtau";
  if(argc >= 2){
    if(strcmp(argv[1], "lscan") == 0)        umode = unfolding::unfoldingMode::LScan;
    else if(strcmp(argv[1], "rhoscan") == 0) umode = unfolding::unfoldingMode::RhoScan;
  }


  // ---------------------------------------------------------------------------
  // read binning schemes in XML format
  TDOMParser parser;
  TString binning_xml = "Binning.xml";
  //
  Int_t error=parser.ParseFile(binning_xml);
  if(error) cout<<"error="<<error<<" from TDOMParser\n";
  TXMLDocument const *XMLdocument=parser.GetXMLDocument();
  binning_rec = TUnfoldBinningXML::ImportXML(XMLdocument,"binning_rec");
  binning_gen = TUnfoldBinningXML::ImportXML(XMLdocument,"binning_gen");
  if(!binning_rec) cout<<"could not read 'rec' binning\n";
  if(!binning_gen) cout<<"could not read 'gen' binning\n";

  // ---------------------------------------------------------------------------
  // read hists
  TFile *hist_file=new TFile("EEEC_Histograms.root", "READ");
  TH1F* input = (TH1F*) hist_file->Get("rec__pseudodata");
  TH2F* response_matrix_pseudo = (TH2F*) hist_file->Get("matrix__pseudodata");
  TH2F* response_matrix = (TH2F*) hist_file->Get("matrix__ttbar");
  TH1F* rec_mc = (TH1F*) hist_file->Get("rec__ttbar");

  double offset_scale = input->Integral()/rec_mc->Integral();

  // ---------------------------------------------------------------------------
  // unfolding
  unfolding my_unfolder(input, response_matrix, binning_rec, binning_gen);
  my_unfolder.set_offset_scale(offset_scale);
  my_unfolder.set_unfolding_mode(umode);
  my_unfolder.set_fixed_tau(0.0);
  my_unfolder.unfold();
  TH1F* h_unfolded = my_unfolder.get_unfolded();
  TH2F* h_unfolded_2D = my_unfolder.get_unfolded_2D();

  // ---------------------------------------------------------------------------
  // unfolding of mc with itself
  unfolding my_unfolder_self_ttbar(rec_mc, response_matrix, binning_rec, binning_gen);
  my_unfolder_self_ttbar.set_offset_scale(1.0);
  my_unfolder_self_ttbar.set_unfolding_mode(unfolding::unfoldingMode::FixedTau);
  my_unfolder_self_ttbar.set_fixed_tau(0.0);
  my_unfolder_self_ttbar.unfold();
  TH1F* h_unfolded_self_ttbar = my_unfolder_self_ttbar.get_unfolded();
  TH2F* h_unfolded_2D_self_ttbar = my_unfolder_self_ttbar.get_unfolded_2D();

  // ---------------------------------------------------------------------------
  // unfolding of pseudodata with itself
  unfolding my_unfolder_self_pseudo(input, response_matrix_pseudo, binning_rec, binning_gen);
  my_unfolder_self_pseudo.set_offset_scale(1.0);
  my_unfolder_self_pseudo.set_unfolding_mode(unfolding::unfoldingMode::FixedTau);
  my_unfolder_self_pseudo.set_fixed_tau(0.0);
  my_unfolder_self_pseudo.unfold();
  TH1F* h_unfolded_self_pseudo = my_unfolder_self_pseudo.get_unfolded();
  TH2F* h_unfolded_2D_self_pseudo = my_unfolder_self_pseudo.get_unfolded_2D();

  // ---------------------------------------------------------------------------
  // save result
  TString filename = "EEEC_Result.root";
  TFile* outputFile = new TFile(filename,"recreate");
  outputFile->cd();
  h_unfolded->Write("unfolded");
  h_unfolded_2D->Write("unfolded_2D");
  h_unfolded_self_ttbar->Write("unfolded_self_ttbar");
  h_unfolded_2D_self_ttbar->Write("unfolded_2D_self_ttbar");
  h_unfolded_self_pseudo->Write("unfolded_self_pseudodata");
  h_unfolded_2D_self_pseudo->Write("unfolded_2D_self_pseudodata");
  outputFile->Close();
  cout << "Wrote histograms to " << filename << endl;

  return 0;
}
