#include <iostream>
#include <TString.h>
#include <TH1.h>
#include <TH2.h>
#include <TTree.h>
#include <TFile.h>


using namespace std;

struct SampleHistogram {
  const TString name;
  const bool isData;
  TTree* tree;
  TH2F* matrix;
  TH1F* rec;
  TH1F* rec_weighted;
  TH1F* gen;
  TH1F* gen_weighted;
  TH1F* purity;
  TH1F* stability;
  TH1F* mtop;

  SampleHistogram(const TString n, const bool d, TTree* t);
  void writeToFile(TFile*);
  void scaleHists(double);

};
