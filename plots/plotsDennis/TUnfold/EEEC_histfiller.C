#include "EEEC_histfiller.h"
#include "helpers.h"


int main(int argc, char* argv[]){
  if(argc >= 2){
    if(strcmp(argv[1], "small") == 0){
      cout << "Small option selected" << endl;
      Nentries_max = 100000;
    }
  }


  // switch on histogram errors
  TH1::SetDefaultSumw2();

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

  // get distributions from measurement phase space and sideband regions
  measurement_rec = binning_rec->FindNode("measurement_rec");
  measurement_gen = binning_gen->FindNode("measurement_gen");


  // ---------------------------------------------------------------------------
  // fill histograms
  // (pseudo) data
  cout << "Get pseudo data file" << endl;
  TFile* pseudodata_File=new TFile("/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/results/pseudodata.root", "READ");
  TTree* pseudodata_tree = (TTree*)pseudodata_File->Get("Events");
  SampleHistogram s_pseudo = SampleHistogram("pseudodata", false, pseudodata_tree);
  SampleHistogram s_pseudo_171p5 = SampleHistogram("pseudodata_171p5", false, pseudodata_tree);
  SampleHistogram s_pseudo_173p5 = SampleHistogram("pseudodata_173p5", false, pseudodata_tree);
  fill_hists(s_pseudo);
  fill_hists(s_pseudo_171p5, "BW_reweight_171p5");
  fill_hists(s_pseudo_173p5, "BW_reweight_173p5");

  // ttbar
  TFile *ttbar_File=new TFile("/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/results/mc.root", "READ");
  SampleHistogram s_ttbar = SampleHistogram("ttbar", false, (TTree*)ttbar_File->Get("Events"));
  fill_hists(s_ttbar);

  // ---------------------------------------------------------------------------
  // Scale pseudo data to normalization of ttbar
  double scale_factor = 1.0;
  if(s_pseudo.rec->Integral() > 0) scale_factor = s_ttbar.rec->Integral()/s_pseudo.rec->Integral();
  cout << "Scaling pseudo data with factor " << scale_factor << endl;
  s_pseudo.scaleHists(scale_factor);
  s_pseudo_171p5.scaleHists(scale_factor);
  s_pseudo_173p5.scaleHists(scale_factor);

  // ---------------------------------------------------------------------------
  // Write output file
  TString filename = "EEEC_Histograms.root";
  outputFile=new TFile(filename,"recreate");
  s_pseudo.writeToFile(outputFile);
  s_pseudo_171p5.writeToFile(outputFile);
  s_pseudo_173p5.writeToFile(outputFile);
  s_ttbar.writeToFile(outputFile);
  outputFile->Close();
  cout << "Saved histograms in " << filename << endl;

  return 0;
}

void fill_hists(SampleHistogram& sample, TString BW_weight_name){
  cout << "Filling histograms for sample "+sample.name+" ...\n";

  // setup hists
  TH2F* h_mc_matrix = (TH2F*) TUnfoldBinning::CreateHistogramOfMigrations(binning_gen, binning_rec, "matrix__"+sample.name);
  TH1F* h_rec = (TH1F*) binning_rec->CreateHistogram("rec__"+sample.name);
  TH1F* h_rec_weighted = (TH1F*) binning_rec->CreateHistogram("rec_weighted__"+sample.name, kTRUE, 0, 0, "weight[C]");
  TH1F* h_gen = (TH1F*) binning_gen->CreateHistogram("gen__"+sample.name);
  TH1F* h_gen_weighted = (TH1F*) binning_gen->CreateHistogram("gen_weighted__"+sample.name, kTRUE, 0, 0, "weight[C]");
  TH1F* purity_sameBin = (TH1F*) measurement_gen->CreateHistogram("purity_sameBin__"+sample.name);
  TH1F* purity_all = (TH1F*) measurement_gen->CreateHistogram("purity_all__"+sample.name);
  TH1F* stability_sameBin = (TH1F*) measurement_gen->CreateHistogram("stability_sameBin__"+sample.name);
  TH1F* stability_all = (TH1F*) measurement_gen->CreateHistogram("stability_all__"+sample.name);
  TH1F* h_mtop = new TH1F("mtop__"+sample.name, "", 50, 170., 175.);

  sample.tree->ResetBranchAddresses();
  sample.tree->SetBranchAddress("ievent",&i_event);
  sample.tree->SetBranchAddress("event_weight_gen",&event_weight_gen);
  sample.tree->SetBranchAddress("event_weight_rec",&event_weight_rec);
  sample.tree->SetBranchAddress("zeta_gen",&zeta_gen);
  sample.tree->SetBranchAddress("zeta_rec",&zeta_rec);
  sample.tree->SetBranchAddress("zeta_weight_gen",&zeta_weight_gen);
  sample.tree->SetBranchAddress("zeta_weight_rec",&zeta_weight_rec);
  sample.tree->SetBranchAddress("has_gen_info",&has_gen_info);
  sample.tree->SetBranchAddress("has_rec_info",&has_rec_info);
  sample.tree->SetBranchAddress("jetpt_rec",&jetpt_rec);
  sample.tree->SetBranchAddress("jetpt_gen",&jetpt_gen);
  sample.tree->SetBranchAddress("pass_triplet_top_rec",&pass_triplet_top_rec);
  sample.tree->SetBranchAddress("pass_triplet_top_gen",&pass_triplet_top_gen);
  sample.tree->SetBranchAddress("mtop",&mtop);
  if(BW_weight_name != "none"){
    cout << "  - using weight " << BW_weight_name << endl;
    sample.tree->SetBranchAddress(BW_weight_name, &BW_factor);
  }
  sample.tree->SetBranchStatus("*",1);

  Double_t rec_weight;
  Double_t gen_weight;
  Double_t w_central;
  Double_t w_nogen;
  Double_t w_norec;
  Double_t w_correction;
  Int_t last_event = -1;

  int Nentries = sample.tree->GetEntriesFast();
  double threshold = 0.0;

  for(Int_t iEntry=0; iEntry < Nentries; iEntry++) {
    if(sample.tree->GetEntry(iEntry)<=0) break;
    if(iEntry > threshold*Nentries){
      showLoadingBar(Nentries, iEntry);
      threshold += 0.05;
    }
    if(Nentries_max != -1 && iEntry > Nentries_max) break;

    // Check if the entry is from the same event as last triplet
    bool same_event = false;
    if(last_event == i_event) same_event = true;
    last_event = i_event;

    // Selections
    bool gen_info = false;
    if(has_gen_info == 1) gen_info = true;
    bool rec_info = false;
    if(has_rec_info == 1) rec_info = true;

    bool triplet_top_gen = false;
    if(pass_triplet_top_gen == 1) triplet_top_gen = true;
    bool triplet_top_rec = false;
    if(pass_triplet_top_rec == 1) triplet_top_rec = true;

    // Additional variables
    double pt_weight_gen = jetpt_gen*jetpt_gen/(172.5*172.5);
    double pt_weight_rec = jetpt_rec*jetpt_rec/(172.5*172.5);
    rec_weight = event_weight_rec;
    gen_weight = event_weight_gen;

    // BW reweighting?
    if(BW_weight_name != "none") gen_weight *= BW_factor;

    // get weights for migration matrix
    w_central = rec_weight * gen_weight;
    w_nogen = rec_weight * gen_weight;
    w_norec = gen_weight;
    w_correction = gen_weight * (1 - rec_weight);

    // get global bins
    Int_t recBin = 0;
    if(rec_info){
      if(triplet_top_rec) recBin = measurement_rec->GetGlobalBinNumber(zeta_rec*pt_weight_rec, zeta_weight_rec);
    }
    Int_t genBin = 0;
    if(gen_info){
      if(triplet_top_gen) genBin = measurement_gen->GetGlobalBinNumber(zeta_gen*pt_weight_gen, zeta_weight_gen);
    }
    // get bin numbers based on gen binning
    int genBin_recInfo = 0;
    int genBin_genInfo = 0;
    if(rec_info && triplet_top_rec) genBin_recInfo = measurement_gen->GetGlobalBinNumber(zeta_rec*pt_weight_rec, zeta_weight_rec);
    if(gen_info && triplet_top_gen) genBin_genInfo = measurement_gen->GetGlobalBinNumber(zeta_gen*pt_weight_gen, zeta_weight_gen);

    // Fill Matrix
    if( rec_info &&  gen_info) h_mc_matrix->Fill(genBin, recBin, w_central);
    if( rec_info && !gen_info) h_mc_matrix->Fill(genBin, recBin, w_nogen);
    if(!rec_info &&  gen_info) h_mc_matrix->Fill(genBin, recBin, w_norec);
    if( rec_info &&  gen_info) h_mc_matrix->Fill(genBin,     0., w_correction); // this is needed because events that dont pass rec, have no rec_weight.

    // Fill Rec
    h_rec->Fill(recBin, w_central);
    if(rec_info && triplet_top_rec) h_rec_weighted->Fill(zeta_rec*pt_weight_rec, w_central*zeta_weight_rec);

    // Fill Gen
    h_gen->Fill(genBin, gen_weight);
    if(gen_info && triplet_top_rec) h_gen_weighted->Fill(zeta_gen*pt_weight_gen, gen_weight*zeta_weight_gen);
    if(gen_info && !same_event) h_mtop->Fill(mtop, gen_weight);

    // Fill purity & stability
    // fill all events in a histogram
    stability_all->Fill(genBin_recInfo, w_central);
    purity_all->Fill(genBin_genInfo, w_central);

    // keep track of events that are in same bin
    if(genBin_recInfo == genBin_genInfo){
      stability_sameBin->Fill(genBin_recInfo, w_central);
      purity_sameBin->Fill(genBin_genInfo, w_central);
    }
  }
  TH1F* h_stability = (TH1F*) stability_sameBin->Clone("stability__"+sample.name);
  TH1F* h_purity = (TH1F*) purity_sameBin->Clone("purity__"+sample.name);
  h_stability->Divide(stability_sameBin, stability_all, 1.0, 1.0, "B");
  h_purity->Divide(purity_sameBin, purity_all, 1.0, 1.0, "B");


  sample.matrix          = (TH2F*) h_mc_matrix->Clone();
  sample.rec             = (TH1F*) h_rec->Clone();
  sample.rec_weighted    = (TH1F*) h_rec_weighted->Clone();
  sample.gen             = (TH1F*) h_gen->Clone();
  sample.gen_weighted    = (TH1F*) h_gen_weighted->Clone();
  sample.purity          = (TH1F*) h_purity->Clone();
  sample.stability       = (TH1F*) h_stability->Clone();
  sample.mtop            = (TH1F*) h_mtop->Clone();


  cout << "Done." << endl;
  return;
}
