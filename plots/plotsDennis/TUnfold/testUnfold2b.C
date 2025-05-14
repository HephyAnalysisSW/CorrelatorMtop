// Author: Stefan Schmitt
// DESY, July 2016

//  Version 17.9, example of using the SURE method
//

#include <iostream>
#include <map>
#include <cmath>
#include <TMath.h>
#include <TRandom3.h>
#include <TFile.h>
#include <TDirectoryFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <THStack.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TLorentzVector.h>
#include "TUnfoldBinning.h"

// defines signal region (gen level)
double const FIDUCIAL_ETAMAX=3.0;

// selection cuts (reco level
double const MASS_MIN=85.;
double const MASS_MAX=120.;
double const MAIN_ETA_MAX=FIDUCIAL_ETAMAX;
double const SIDE_ETA_MAX=4.0;

using namespace std;

/*
  This file is part of TUnfold.

  TUnfold is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  TUnfold is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with TUnfold.  If not, see <http://www.gnu.org/licenses/>.
*/

///////////////////////////////////////////////////////////////////////
// 
// Test program for the classes TUnfoldDensity and TUnfoldBinning
//
// A toy test of the TUnfold package
//
// This is an example of unfolding a one-dimensional distribution
//   plus nuisance parameters to control background from fakes
//
// The example comprizes several macros
//   testUnfold2a.C   create root files with TTree objects for
//                      signal, background and data
//            -> write files  testUnfold2a_MC.root
//                            testUnfold2a_data.root
//
//   testUnfold2b.C   loop over trees and fill histograms based on the
//                      TUnfoldBinning objects
//            -> read  testUnfold2a_MC.root
//                     testUnfold2a_data.root
//            -> write testUnfold2b_histograms.root
//            -> produce plots testUnfold2b_controlPlots.eps
//            -> produce plots testUnfold2b_migrations.eps
//
//   testUnfold2c.C   run the unfolding
//            -> read  testUnfold2b_histograms.root
//            -> write testUnfold2c_unfolded.root
//            -> produce plots
// 
///////////////////////////////////////////////////////////////////////

void testUnfold2b()
{
   //==================================================================  
   // fill histograms


   TH1::SetDefaultSumw2();
   gStyle->SetOptFit(0);
   gStyle->SetOptStat(0);
   gStyle->SetPadRightMargin(0.06);
   gStyle->SetPadBottomMargin(0.18);
   gStyle->SetPadLeftMargin(0.19);
   gStyle->SetPadTopMargin(0.07);
   int font=42;
   gStyle->SetLabelFont(font,"XYZ");
   gStyle->SetLegendFont(font);
   gStyle->SetStatFont(font);
   gStyle->SetTextFont(font);
   gStyle->SetTitleFont(font,"XYZ");
   gStyle->SetTitleOffset(1.5,"y");
   gStyle->SetTitleOffset(1.2,"x");
   gStyle->SetTitleSize(0.8,"P");
   gStyle->SetTitleSize(0.06,"xy");
   gStyle->SetLabelSize(0.05,"xy");
   gStyle->SetLabelOffset(0.012,"xy");

   TFile *histogramFile=new TFile("testunfold2b_histograms.root","recreate");
   histogramFile->cd();
   // control distributions
   // each control distributions is filled four times
   //   [0] -> data
   //   [1] -> MC signal
   //   [2] -> MC fakes
   //   [3] -> MC background

   // eta(reco) distribution (for fakes sideband definition)
   int nBinEta=50;
   double etaMin=-4.;
   double etaMax=4.;
   TH1 *hist_etarec[4];
   hist_etarec[0]=new TH1D("hist_etarec_data",";#eta_{reco} [GeV]",
                           nBinEta,etaMin,etaMax);
   hist_etarec[1]=new TH1D("hist_etarec_Signal",";#eta_{reco} [GeV]",
                           nBinEta,etaMin,etaMax);
   hist_etarec[2]=new TH1D("hist_etarec_Fakes",";#eta_{reco} [GeV]",
                           nBinEta,etaMin,etaMax);
   hist_etarec[3]=new TH1D("hist_etarec_Bgr",";#eta_{reco} [GeV]",
                           nBinEta,etaMin,etaMax);

   // define PT binnings here
   // coarse PT binning
   static double const binsPtCoarseArray[]={0.,2.,4.,6.,8.,10.,12.5,15.,17.5,20.,25.,30.,35.,40.,45.,50.,70.,100.};
   int sizePtCoarseArray=sizeof(binsPtCoarseArray)/sizeof(double)-1;

   // [0] -> coarse binning
   // [1] -> fine binning
   vector<double> binsPtVector[2];
   for(int i=0;i<=sizePtCoarseArray;i++) {
      double x0=binsPtCoarseArray[i];
      for(int k=0;k<2;k++) binsPtVector[k].push_back(x0);
      if(i<sizePtCoarseArray) {
         // add one extra bin for k=1
         double x1=binsPtCoarseArray[i+1];
         double x2=0.5*(x0+x1);
         //  [x0,x2] [x2,x1]
         binsPtVector[1].push_back(x2);
      }
   }

   vector<double> const &binsPtFine=binsPtVector[1];
   int numBinsPtFine=binsPtFine.size()-1;
   // pt(reco) distributions (fine binning) for main selection
   TH1 *hist_ptrec_main[4];
   hist_ptrec_main[0]=new TH1D("hist_ptrec_main_data",";P_{T,reco} [GeV]",
                               numBinsPtFine,binsPtFine.data());
   hist_ptrec_main[1]=new TH1D("hist_ptrec_main_Signal",";P_{T,reco} [GeV]",
                               numBinsPtFine,binsPtFine.data());
   hist_ptrec_main[2]=new TH1D("hist_ptrec_main_Fakes",";P_{T,reco} [GeV]",
                               numBinsPtFine,binsPtFine.data());
   hist_ptrec_main[3]=new TH1D("hist_ptrec_main_Bgr",";P_{T,reco} [GeV]",
                               numBinsPtFine,binsPtFine.data());

   // (3c) pt(reco) distributions (fine binning) for fakes sidebin
   TH1 *hist_ptrec_sideFakes[4];
   hist_ptrec_sideFakes[0]=new TH1D("hist_ptrec_sideFakes_data",";P_{T,reco} [GeV]",
                                numBinsPtFine,binsPtFine.data());
   hist_ptrec_sideFakes[1]=new TH1D("hist_ptrec_sideFakes_Signal",";P_{T,reco} [GeV]",
                                numBinsPtFine,binsPtFine.data());
   hist_ptrec_sideFakes[2]=new TH1D("hist_ptrec_sideFakes_Fakes",";P_{T,reco} [GeV]",
                                numBinsPtFine,binsPtFine.data());
   hist_ptrec_sideFakes[3]=new TH1D("hist_ptrec_sideFakes_Bgr",";P_{T,reco} [GeV]",
                                numBinsPtFine,binsPtFine.data());

   // input distribution for unfolding
   // the binning is similar to the "pt" binning,
   // but there is one extra bin for normalizing the fakes
   //   and there is also the overflow bin
   // These bining schemes are handled by the class  "TUnfoldBinning"
   TUnfoldBinning *binningRec[2],*binningGen[2];
   TString const  binningType[2]={"Coarse","Fine"};
   for(int k=0;k<2;k++) {
      binningRec[k]=new TUnfoldBinning(binningType[k]+"Reco");
      binningRec[k]->AddBinning(new TUnfoldBinning("sideBinFakes",1));
      // add binning without underflow (false) but with overflow (true)
      binningRec[k]->AddBinning(new TUnfoldBinning("mainSelection"))
         ->AddAxis("P_{T,reco}",binsPtVector[k].size()-1,binsPtVector[k].data(),
                   false,true);
      binningGen[k]=new TUnfoldBinning(binningType[k]+"Gen");
      binningGen[k]->AddBinning(new TUnfoldBinning("fakes",1));
      binningGen[k]->AddBinning(new TUnfoldBinning("signal"))
         ->AddAxis("P_{T,gen}",binsPtVector[k].size()-1,binsPtVector[k].data(),
                   false,true);
   }

   // book distributions 
   // data (coarse/fine binning)
   TH1 *hist_unfoldingReco_data[2];
   // BGR reco (coarse/fine binning)
   TH1 *hist_unfoldingReco_bgr[2];
   // MC reco (coarse/fine binning)
   TH1 *hist_unfoldingReco_MC[2];
   // MC truth (coarse/fine binning)
   TH1 *hist_unfoldingGen_MC[3];
   for(int k=0;k<2;k++) {
      hist_unfoldingReco_data[k]=binningRec[k]->CreateHistogram
         ("hist_unfoldingReco"+binningType[k]+"_data");
      hist_unfoldingReco_MC[k]=binningRec[k]->CreateHistogram
         ("hist_unfoldingReco"+binningType[k]+"_MC");
      hist_unfoldingReco_bgr[k]=binningRec[k]->CreateHistogram
         ("hist_unfoldingReco"+binningType[k]+"_bgr");
      hist_unfoldingGen_MC[k]=binningGen[k]->CreateHistogram
         ("hist_unfoldingGen"+binningType[k]+"_MC");
   }
   // matrix of migrations
   // six matrices are considered
   //       reco        gen
   // [0]  0=coarse  0=coarse
   // [1]  1=fine    0=coarse
   // [2]  1=fine    1=fine
   //
   // the index 0..3 is given by: k= iGen+iRec*(iRec+1)/2 
   //
   TH2 *hist_migration_MC[3];
   for(int iGen=0;iGen<2;iGen++) {
      for(int iRec=iGen;iRec<2;iRec++) {
         int k=iGen+iRec*(iRec+1)/2;
         hist_migration_MC[k]=
            TUnfoldBinning::CreateHistogramOfMigrations
            (binningGen[iGen],binningRec[iRec],
             "hist_migration"+binningType[iGen]+binningType[iRec]+"_MC");
      }
   }

   //
   // fill histograms
   const char *fileName[2]={"testUnfold2a_data.root","testUnfold2a_MC.root"};
   bool isMC[2]={false,true};

   for(int ifile=0;ifile<2;ifile++) {
      cout<<"loop over: "<<fileName[ifile]<<"\n";
      TFile *file=new TFile(fileName[ifile]);
      Float_t etaRec,ptRec,mRec,weightRec;
      Int_t hasTriggered;
      Float_t ptGen,etaGen,weightGen;
      Int_t isSignal;

      TTree *tree=0;
      file->GetObject("event",tree);
      if(!tree) {
         delete file;
         continue;
      }

      tree->SetBranchAddress("etaRec",&etaRec);
      tree->SetBranchAddress("ptRec",&ptRec);
      tree->SetBranchAddress("mRec",&mRec);
      tree->SetBranchAddress("weightRec",&weightRec);
      tree->SetBranchAddress("hasTriggered",&hasTriggered);
      if(isMC[ifile]) {
         tree->SetBranchAddress("ptGen",&ptGen);
         tree->SetBranchAddress("etaGen",&etaGen);
         tree->SetBranchAddress("weightGen",&weightGen);
         tree->SetBranchAddress("isSignal",&isSignal);
      }
      cout<<"number of events: "<<tree->GetEntries()<<"\n";

      int printEvent=0;

      for(Long_t ievent=0;ievent<tree->GetEntries();ievent++) {
         tree->GetEvent(ievent);
         // MC truth classification
         bool isFake=false,isBackground=false,isFiducial=false;
         if(isMC[ifile]) {
            if(isSignal) {
               // apply truth phase-space cuts
               isFiducial=TMath::Abs(etaGen)<FIDUCIAL_ETAMAX;
               isFake= !isFiducial;
            } else {
               isBackground=true;
            }
         }
         // reconstructed event classification
         bool massCut=false;
         bool etaCutMain=false,etaCutSide=false;
         if(hasTriggered) {
            // cuts on reco level, defining main selection and sidebands
            massCut=(mRec>=MASS_MIN)&&(mRec<MASS_MAX);
            etaCutMain=TMath::Abs(etaRec)<MAIN_ETA_MAX;
            etaCutSide=(TMath::Abs(etaRec)<SIDE_ETA_MAX)&&!etaCutMain;
         }
         // index of control histogram
         int iControl=0; // data
         if(isFiducial) iControl=1;
         else if(isFake) iControl=2;
         else if(isBackground) iControl=3;
         
         if(printEvent) {
            cout<<" gen: "<<isFiducial<<isFake<<isBackground
                <<" rec: "<<massCut<<etaCutMain<<etaCutSide
                <<" control: "<<iControl;
         }
         // fill control distributions
         if(massCut) {
            hist_etarec[iControl]->Fill(etaRec,weightRec);
         }
         if(etaCutMain && massCut) {
            hist_ptrec_main[iControl]->Fill(ptRec,weightRec);
         }
         if(etaCutSide && massCut) {
            hist_ptrec_sideFakes[iControl]->Fill(ptRec,weightRec);
         }

         // fill unfolding histograms with the help of the binning schemes
         
         // first,calculate reconstructed and generated bin numbers
         int recoBin[2];
         if(printEvent) cout<<" reco [";
         for(int irec=0;irec<2;irec++) {
            recoBin[irec]=0; // not in any of the reco bins
            if(etaCutMain && massCut) {
               // main selection, bin number depends on ptRec
               TUnfoldBinning const *where=
                  binningRec[irec]->FindNode("mainSelection");
               recoBin[irec]=where->GetGlobalBinNumber(ptRec);
            } else if(etaCutSide && massCut) {
               // side bin enriched in fakes
               TUnfoldBinning const *where=
                  binningRec[irec]->FindNode("sideBinFakes");
               recoBin[irec]=where->GetStartBin();
            }
            if(printEvent) cout<<" "<<recoBin[irec];
         }
         int genBin[2];
         if(printEvent) cout<<" ] gen [";
         for(int igen=0;igen<2;igen++) {
            genBin[igen]=0; // not in any generator bin (data? background?)
            if(isFiducial) {
               // binning in ptGen
               TUnfoldBinning const *where=
                  binningGen[igen]->FindNode("signal");
               genBin[igen]=where->GetGlobalBinNumber(ptGen);
            } else if(isFake) {
               TUnfoldBinning const *where=
                  binningGen[igen]->FindNode("fakes");
               genBin[igen]=where->GetStartBin();
            }
            if(printEvent) cout<<" "<<genBin[igen];
         }
         if(printEvent) cout<<" ]\n";
         
         // fill reco histograms
         for(int ireco=0;ireco<2;ireco++) {
            if(isMC[ifile]) {
               if(isBackground) {
                  hist_unfoldingReco_bgr[ireco]->Fill(recoBin[ireco],weightRec);
               } else {
                  hist_unfoldingReco_MC[ireco]->Fill(recoBin[ireco],weightRec);
               }
            } else {
               hist_unfoldingReco_data[ireco]->Fill(recoBin[ireco],weightRec);
            }
         }
         if(isMC[ifile] && !isBackground) {
            // fill truth histograms
            for(int igen=0;igen<2;igen++) {
               hist_unfoldingGen_MC[igen]->Fill(genBin[igen],weightGen);
               for(int irec=igen;irec<2;irec++) {
                  // fill matrix
                  int k=igen+irec*(irec+1)/2;
                  hist_migration_MC[k]->Fill(genBin[igen],recoBin[irec],
                                             weightRec);
                  // count fraction of events which have a different weight on truth and reco (in reco underflow bin)
                  // this is required for TUnfold to function properly
                  hist_migration_MC[k]->Fill(genBin[igen],0.,
                                             weightGen-weightRec);
               }
            }
         }
      }
      delete tree;
      delete file;
   }

   ///
   // save histograms etc
   histogramFile->cd();
   //
   // save binning objects
   for(int igen=0;igen<2;igen++) {
      binningGen[igen]->Write();
   }
   for(int irec=0;irec<2;irec++) {
      binningRec[irec]->Write();
   }
   //
   // control distributions (4 histograms each)
   for(int i=0;i<4;i++) {
      hist_etarec[i]->Write();
      hist_ptrec_main[i]->Write();
      hist_ptrec_sideFakes[i]->Write();
   }
   // unfolding input and truth distributions (3 histograms each)
   for(int i=0;i<2;i++) {
      hist_unfoldingReco_data[i]->Write();
      hist_unfoldingReco_bgr[i]->Write();
      hist_unfoldingReco_MC[i]->Write();
      hist_unfoldingGen_MC[i]->Write();
   }
   // migration matrix (6 histograms)
   for(int i=0;i<3;i++) {
      hist_migration_MC[i]->Write();
   }

   // calculate bin purities for the three binning schemes
   TH1 *hist_purity[2];
   for(int iRec=0;iRec<2;iRec++) {
      int k=iRec+iRec*(iRec+1)/2;
      hist_purity[iRec]=binningRec[iRec]->CreateHistogram
         ("hist_purity"+binningType[iRec]);
      for(int binRec=0;binRec<=hist_purity[iRec]->GetNbinsX()+1;binRec++) {
         double sum=0.;
         for(int binGen=0;binGen<=hist_purity[iRec]->GetNbinsX()+1;binGen++) {
            sum += hist_migration_MC[k]->GetBinContent(binGen,binRec);
         }
         double p=0.;
         if(sum>0.0) {
            p=hist_migration_MC[k]->GetBinContent(binRec,binRec)/sum;
         }
         hist_purity[iRec]->SetBinContent(binRec,p);
      }
      // set bin labels
      hist_purity[iRec]->GetXaxis()->SetBinLabel(binningRec[iRec]->FindNode("sideBinFakes")->GetStartBin(),"fakes side bin");
      TUnfoldBinning const *ptBinning=
         binningRec[iRec]->FindNode("mainSelection");
      for(int ptBin=ptBinning->GetStartBin();ptBin<ptBinning->GetEndBin();
          ptBin++) {
         TString  label(ptBinning->GetBinName(ptBin));
         label.Remove(0,label.Last('['));
         label.Remove(label.Length()-1,1);
         hist_purity[iRec]->GetXaxis()->SetBinLabel(ptBin,label);
      }
      hist_purity[iRec]->SetTitle(";;bin purity");      
      hist_purity[iRec]->Write();
   }

   // calculate efficiency and matrix of probabilities for the six matrices
   TH1 *hist_efficiency[2];
   TH2 *hist_probability[3];
   for(int iGen=0;iGen<2;iGen++) {
      TUnfoldBinning const *ptBinning=
         binningGen[iGen]->FindNode("signal");
      for(int iRec=iGen;iRec<2;iRec++) {
         int k=iGen+iRec*(iRec+1)/2;
         hist_probability[k]=
            TUnfoldBinning::CreateHistogramOfMigrations
            (binningGen[iGen],binningRec[iRec],
             "hist_probability"+binningType[iGen]+binningType[iRec]+"_MC");
         for(int binGen=0;binGen<=hist_probability[k]->GetNbinsX()+1;binGen++) {
            double sum=0.;
            for(int binRec=0;binRec<=hist_probability[k]->GetNbinsY()+1;
                binRec++) {
               sum += hist_migration_MC[k]->GetBinContent(binGen,binRec);
            }
            if(sum>0.) {
               for(int binRec=0;binRec<=hist_probability[k]->GetNbinsY()+1;
                   binRec++) {
                  hist_probability[k]->SetBinContent
                     (binGen,binRec,
                      hist_migration_MC[k]->GetBinContent(binGen,binRec)/sum);
               }
            }
         }
         int skip=0;
         for(int ptBin=ptBinning->GetEndBin()-1;ptBin>=ptBinning->GetStartBin();
             ptBin--) {
            TString label=".";
            if(!skip) {
               label=ptBinning->GetBinName(ptBin);
               label.Remove(0,label.Last('['));
               label.Remove(label.Length()-1,1);
            }
            hist_probability[k]->GetXaxis()->SetBinLabel(ptBin,label);
            skip++;
            if(skip==4) skip=0;
         }
         TUnfoldBinning const *ptBinningRec=
            binningRec[iRec]->FindNode("mainSelection");
         skip=0;
         int blank=1;
         for(int ptBin=ptBinningRec->GetEndBin()-1;
             ptBin>=ptBinningRec->GetStartBin();
             ptBin--) {
            TString label=".";
            if(!skip) {
               label=ptBinningRec->GetBinName(ptBin);
               label.Remove(0,label.Last('['));
               label.Remove(label.Length()-1,1);
            }
            hist_probability[k]->GetYaxis()->SetBinLabel(ptBin,label);
            skip++;
            if(((skip==4)&&(!blank))||(skip==12)) {
               skip=0;
               blank=0;
            }
         }


         hist_probability[k]->GetXaxis()->SetBinLabel
            (binningGen[iGen]->FindNode("fakes")->GetStartBin(),"fakes");
         hist_probability[k]->GetYaxis()->SetBinLabel
            (binningRec[iRec]->FindNode("sideBinFakes")->GetStartBin(),"sideband");
         hist_probability[k]->SetTitle(";P_{T,gen} bins;P_{T,rec} bins;probability");
         hist_probability[k]->Write();
      }
      // efficiency
      hist_efficiency[iGen]=binningGen[iGen]->CreateHistogram
         ("hist_efficiency"+binningType[iGen]);
      int k=iGen+iGen*(iGen+1)/2;
      for(int binGen=0;binGen<=hist_probability[k]->GetNbinsX()+1;binGen++) {
         double sum0=0.;
         double sum1=0.;
         for(int binRec=0;binRec<=hist_probability[k]->GetNbinsY()+1;
             binRec++) {
            double c=hist_migration_MC[k]->GetBinContent(binGen,binRec);
            sum0+=c;
            if((binRec>0)&&(binRec<=hist_probability[k]->GetNbinsY())) {
               sum1+=c;
            }
         }
         if(sum0>0.0) {
            hist_efficiency[iGen]->SetBinContent(binGen,sum1/sum0);
         }
      }
      hist_efficiency[iGen]->GetXaxis()->SetBinLabel
         (binningGen[iGen]->FindNode("fakes")->GetStartBin(),"fakes");
      //for(int ptBin=ptBinning->GetStartBin();ptBin<ptBinning->GetEndBin();
      //    ptBin++) {
      int skip=0;
      skip=0;
      for(int ptBin=ptBinning->GetEndBin()-1;ptBin>=ptBinning->GetStartBin();
          ptBin--) {
         TString label=".";
         if(!skip) {
            label=ptBinning->GetBinName(ptBin);
            label.Remove(0,label.Last('['));
            label.Remove(label.Length()-1,1);
         }
         hist_efficiency[iGen]->GetXaxis()->SetBinLabel(ptBin,label);
         skip++;
         if(skip==4) skip=0;
      }
      hist_efficiency[iGen]->SetTitle(";P_{T,gen} bins;efficiency");

      hist_efficiency[iGen]->Write();
   }


   // produce plots
   TCanvas *can=new TCanvas("control","control",600,300);
   can->Divide(2,1);
   can->cd(1);
   int dataCol=kRed+4;
   double dataMsize=0.7;
   hist_ptrec_main[0]->SetLineColor(dataCol);
   hist_ptrec_main[0]->SetMarkerColor(dataCol);
   hist_ptrec_main[0]->SetMarkerStyle(20);
   hist_ptrec_main[0]->SetMarkerSize(dataMsize);
   hist_ptrec_sideFakes[0]->SetLineColor(dataCol);
   hist_ptrec_sideFakes[0]->SetMarkerColor(dataCol);
   hist_ptrec_sideFakes[0]->SetMarkerStyle(20);
   hist_ptrec_sideFakes[0]->SetMarkerSize(dataMsize);

   for(int k=1;k<4;k++) {
      int c=kMagenta;
      if(k==2) c=kBlue;
      if(k==3) c=kCyan;
      hist_ptrec_main[k]->SetFillStyle(1001);
      hist_ptrec_main[k]->SetLineColor(c-3);
      hist_ptrec_main[k]->SetFillColor(c);
      hist_ptrec_sideFakes[k]->SetFillStyle(1001);
      hist_ptrec_sideFakes[k]->SetLineColor(c-3);
      hist_ptrec_sideFakes[k]->SetFillColor(c);

   }

   THStack *mcMain=new THStack("mc",";P_{T,rec} [GeV];events");
   mcMain->Add( hist_ptrec_main[3]); // background
   mcMain->Add( hist_ptrec_main[2]); // fakes
   mcMain->Add( hist_ptrec_main[1]); // signal
   mcMain->SetMaximum
      (1.1*TMath::Max(mcMain->GetMaximum(),hist_ptrec_main[0]->GetMaximum()));
   mcMain->Draw("HIST");
   hist_ptrec_main[0]->Draw("SAME E");
   TLegend *legend1=new TLegend(0.3,0.6,0.9,0.9,"signal region");
   legend1->SetBorderSize(0);
   legend1->SetFillStyle(0);
   legend1->SetTextSize(0.04);
   legend1->AddEntry(hist_ptrec_main[0],"\"Data\"","p");
   legend1->AddEntry(hist_ptrec_main[1],"MC signal","lf");
   legend1->AddEntry(hist_ptrec_main[2],"MC fakes","lf");
   legend1->AddEntry(hist_ptrec_main[3],"MC background","lf");
   legend1->Draw();

   can->cd(2);
   THStack *mcSideF=new THStack("mc",";P_{T,rec} [GeV];events");
   mcSideF->Add( hist_ptrec_sideFakes[3]); // background
   mcSideF->Add( hist_ptrec_sideFakes[2]); // fakes
   mcSideF->Add( hist_ptrec_sideFakes[1]); // signal
   mcSideF->Draw("HIST");
   mcSideF->SetMaximum
      (1.1*TMath::Max(mcSideF->GetMaximum(),hist_ptrec_sideFakes[0]->GetMaximum()));
   hist_ptrec_sideFakes[0]->Draw("SAME E");
   TLegend *legend3=new TLegend(0.3,0.8,0.9,0.9,"fakes sideband");
   legend3->SetBorderSize(0);
   legend3->SetFillStyle(0);
   legend3->SetTextSize(0.04);
   legend3->Draw();

   can->SaveAs("testunfold2b_controlPlots.eps");

   // show purity and response matrix
   TCanvas *can2=new TCanvas("genrec","genrec",600,300);
   can2->Divide(2,1);
   can2->cd(1);
   hist_efficiency[0]->SetLineColor(kRed);
   hist_efficiency[0]->GetYaxis()->SetTitle("");
   hist_efficiency[0]->GetYaxis()->SetRangeUser(0.0,1.0);
   hist_efficiency[0]->Draw();
   hist_purity[0]->SetLineColor(kBlue);
   hist_purity[0]->Draw("SAME");
   TLegend *legend4=new TLegend(0.3,0.8,0.9,0.9);
   legend4->SetBorderSize(0);
   legend4->SetFillStyle(0);
   legend4->SetTextSize(0.04);
   legend4->AddEntry(hist_efficiency[0],"effiency");
   legend4->AddEntry(hist_purity[0],"purity");
   legend4->Draw();

   can2->cd(2);
   gPad->SetLeftMargin(0.15);
   gPad->SetRightMargin(0.17);
   hist_probability[1]->GetYaxis()->SetTitleOffset(0.9);
   hist_probability[1]->Draw("COLZ");
   //can2->cd(3);
   //hist_purity[0]->Draw();

   can2->SaveAs("testunfold2b_migrations.eps");


   delete histogramFile;

}
