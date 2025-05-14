#include "SampleHistogram.h"


SampleHistogram::SampleHistogram(const TString n, const bool d, TTree* t): name(n), isData(d), tree(t){}

void SampleHistogram::writeToFile(TFile* outputFile){
  outputFile->cd();
  rec->Write("rec__"+name);
  gen->Write("gen__"+name);
  rec_weighted->Write("rec_weighted__"+name);
  gen_weighted->Write("gen_weighted__"+name);
  matrix->Write("matrix__"+name);
  purity->Write("purity__"+name);
  stability->Write("stability__"+name);
  mtop->Write("mtop__"+name);
  cout << "Wrote histograms of " << name << endl;
  return;
}

void SampleHistogram::scaleHists(double factor){
  rec->Scale(factor);
  gen->Scale(factor);
  rec_weighted->Scale(factor);
  gen_weighted->Scale(factor);
  matrix->Scale(factor);
  mtop->Scale(factor);
  return;
}
