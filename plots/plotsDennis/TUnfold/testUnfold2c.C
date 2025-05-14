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
#include <TProfile.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <THStack.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TLine.h>
#include <TLorentzVector.h>
#include "TUnfoldDensity.h"
#include "TUnfoldIterativeEM.h"

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
//            -> produce plots
//
//   testUnfold2c.C   run the unfolding
//            -> read  testUnfold2b_histograms.root
//            -> write testUnfold2c_unfolded.root
//            -> produce plots
// 
///////////////////////////////////////////////////////////////////////

static TH1 *generatePoissonToy(TH1 *base,int ntoy);
static void analyzeToy(TH1 const *hist_toy,
                       TH1 const *hist_truth,TProfile *&prof_pull,
                       TH1 *&coverage,int nToyTotal);

void testUnfold2c()
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

   TFile *inputFile=new TFile("testunfold2b_histograms.root");

   TFile *outputFile=new TFile("testunfold2c_unfolded.root","recreate");
   outputFile->cd();
   TDirectoryFile *inputDir=new TDirectoryFile("input","input");
   inputDir->cd();

   // read input histograms
   //   MC matrix rec vs gen
   TUnfoldBinning  *binningCoarseGen,*binningFineReco;
   TH1 *hist_unfoldingRecoFine_data;
   TH1 *hist_unfoldingRecoFine_MC;
   TH1 *hist_unfoldingRecoFine_bgr;
   TH1 *hist_unfoldingGenCoarse_MC;
   TH2 *hist_migrationCoarseFine_MC;
   
   inputFile->GetObject("CoarseGen",binningCoarseGen);
   inputFile->GetObject("FineReco",binningFineReco);
   inputFile->GetObject("hist_unfoldingRecoFine_data",hist_unfoldingRecoFine_data);
   inputFile->GetObject("hist_unfoldingRecoFine_MC",hist_unfoldingRecoFine_MC);
   inputFile->GetObject("hist_unfoldingRecoFine_bgr",hist_unfoldingRecoFine_bgr);
   inputFile->GetObject("hist_unfoldingGenCoarse_MC",hist_unfoldingGenCoarse_MC);
   inputFile->GetObject("hist_migrationCoarseFine_MC",hist_migrationCoarseFine_MC);

   binningCoarseGen->Write();
   binningFineReco->Write();
   hist_unfoldingRecoFine_data->Write();
   hist_unfoldingRecoFine_MC->Write();
   hist_unfoldingGenCoarse_MC->Write();
   hist_migrationCoarseFine_MC->Write();

   outputFile->cd();
   TDirectoryFile *tunfoldDir=new TDirectoryFile("tunfold","tunfold");
   tunfoldDir->cd();

   //========================================================
   // outer loop: unfold data, then unfold toys from unfolded data

   // histograms to draw toys from
   TH1 *hist_toybase_noRegularisation=0;
   TH1 *hist_toybase_TikhonovSURE=0;
   TH1 *hist_toybase_TikhonovLCurve=0;
   TH1 *hist_toybase_IterativeSURE=0;
   TH1 *hist_toybase_IterativeFixed=0;

   // analysis of toys
   TH1 *hist_unfolded_noRegularisation=0;
   TH1 *hist_unfolded_TikhonovSURE=0;
   TH1 *hist_unfolded_TikhonovLCurve=0;
   TH1 *hist_unfolded_IterativeSURE=0;
   TH1 *hist_unfolded_IterativeFixed=0;
   TProfile *prof_pull_noRegularisation=0;
   TProfile *prof_pull_TikhonovSURE=0;
   TProfile *prof_pull_TikhonovLCurve=0;
   TProfile *prof_pull_IterativeSURE=0;
   TProfile *prof_pull_IterativeFixed=0;
   TH1 *hist_coverage_noRegularisation=0;
   TH1 *hist_coverage_TikhonovSURE=0;
   TH1 *hist_coverage_TikhonovLCurve=0;
   TH1 *hist_coverage_IterativeSURE=0;
   TH1 *hist_coverage_IterativeFixed=0;

   // auxillary output
   // Tikhonov SURE scan
   TGraph *graph_logTauSURE_TikhonovSURE,*graph_dfChi2A_TikhonovSURE;
   int iBest_TikhonovSURE=-1;
   // Tikhonov, minimum L-Curve curvature
   TGraph *graph_LCurve_TikhonovLCurve;
   TSpline *spline_Curvature_TikhonovLCurve;
   int iBest_TikhonovLCurve=-1;
   double tauBest_TikhonovLCurve=-1.,DF_TikhonovLCurve=-1.;
   // Iterative SURE scan
   TGraph *graph_SURE_IterativeSURE,*graph_DFdeviance_IterativeSURE;
   int iBest_IterativeSURE=-1;

   double biasScale=1.0;

   int MAXTOY=300;

   int NPOINT_TikhonovSURE=50;
   int NPOINT_TikhonovLCurve=50;
   int NUM_FIXED_ITERATION=4;
   int NITER_Iterative=100;

   for(int itoy=0;itoy<=MAXTOY;itoy++) {
      cout<<"================== itoy="<<itoy<<" ==========================\n";
      TH1 *input;
      int iBest;
      //============================= (1) ============================
      // chi**2 fit without regularisation
      {
      if(itoy==0) {
         input=hist_unfoldingRecoFine_data;
      } else {
         input=generatePoissonToy
            (hist_toybase_noRegularisation,itoy);
      }

      TUnfoldDensity tunfoldNoRegularisation
         (hist_migrationCoarseFine_MC,TUnfoldDensity::kHistMapOutputHoriz,
          TUnfoldDensity::kRegModeSize,TUnfoldDensity::kEConstraintNone,
          TUnfoldDensity::kDensityModeNone,binningCoarseGen,
          binningFineReco);
      tunfoldNoRegularisation.SubtractBackground(hist_unfoldingRecoFine_bgr,"bgr");
      tunfoldNoRegularisation.SetInput(input,biasScale);
      tunfoldNoRegularisation.DoUnfold(0.0);
      //
      if(itoy==0) {
         // all unfolded bins (here including fakes normalisation)
         hist_unfolded_noRegularisation=
            tunfoldNoRegularisation.GetOutput
            ("hist_unfolded_noRegularisation",";bin",0,0,false);
         hist_unfolded_noRegularisation->Write();
         // unfolding result folded back -> for generating toys
         hist_toybase_noRegularisation=tunfoldNoRegularisation.GetFoldedOutput
            ("hist_toybase_noRegularisation",";bin",0,0,false,true);
         hist_toybase_noRegularisation->Write();
         // unfolding result, signal only
         TH1 *hist_PTunfolded_noRegularisation=
            tunfoldNoRegularisation.GetOutput
            ("hist_PTunfolded_noRegularisation",
             "P_{T,unfolded} [GeV]","signal");
         hist_PTunfolded_noRegularisation->Write();
      } else {
         // analyze toys
         TH1 *hist_unfolded_noRegularisation_toy=
            tunfoldNoRegularisation.GetOutput
            (TString::Format("hist_unfolded_noRegularisation_toy%d",itoy),
             ";bin",0,0,false);
         analyzeToy(hist_unfolded_noRegularisation_toy,
                    hist_unfolded_noRegularisation,
                    prof_pull_noRegularisation,
                    hist_coverage_noRegularisation,MAXTOY);
      }
      }
      //=========================================================
      // Tikhonov, minimum of the SURE variable
      {
      if(itoy==0) {
         input=hist_unfoldingRecoFine_data;
      } else {
         input=generatePoissonToy(hist_toybase_TikhonovSURE,itoy);
      }

      TUnfoldDensity tunfoldTikhonovSURE
         (hist_migrationCoarseFine_MC,TUnfoldDensity::kHistMapOutputHoriz,
          TUnfoldDensity::kRegModeSize,TUnfoldDensity::kEConstraintNone,
          TUnfoldDensity::kDensityModeNone,binningCoarseGen,
          binningFineReco);
      tunfoldTikhonovSURE.SubtractBackground(hist_unfoldingRecoFine_bgr,"bgr");
      tunfoldTikhonovSURE.SetInput(input,biasScale);
      iBest=tunfoldTikhonovSURE.ScanSURE
         (NPOINT_TikhonovSURE,0.,0.,
          (itoy==0) ? &graph_logTauSURE_TikhonovSURE : 0,
          (itoy==0) ? &graph_dfChi2A_TikhonovSURE : 0,
          0);
      if(itoy==0) {
         iBest_TikhonovSURE=iBest;
         // all unfolded bins (here including fakes normalisation)
         hist_unfolded_TikhonovSURE=
            tunfoldTikhonovSURE.GetOutput("hist_unfolded_TikhonovSURE",
                                          ";bin",0,0,false);
         hist_unfolded_TikhonovSURE->Write();
         // unfolding result folded back -> for generating toys
         hist_toybase_TikhonovSURE=tunfoldTikhonovSURE.GetFoldedOutput
            ("hist_toybase_TikhonovSURE",";bin",0,0,false,true);
         hist_toybase_TikhonovSURE->Write();
         // unfolding result, signal only
         TH1 *hist_PTunfolded_TikhonovSURE=
            tunfoldTikhonovSURE.GetOutput("hist_PTunfolded_TikhonovSURE",
                                          "P_{T,unfolded} [GeV]","signal");
         hist_PTunfolded_TikhonovSURE->Write();
         // save auxillary plots
         graph_logTauSURE_TikhonovSURE->Write("graph_logTauSURE_TikhonovSURE");
         graph_dfChi2A_TikhonovSURE->Write("graph_dfChi2A_TikhonovSURE");
      } else {
         // analyze toys
         TH1 *hist_unfolded_TikhonovSURE_toy=
            tunfoldTikhonovSURE.GetOutput
            (TString::Format("hist_PTunfolded_TikhonovSURE_toy%d",itoy),
             ";bin",0,0,false);
         analyzeToy(hist_unfolded_TikhonovSURE_toy,
                    hist_unfolded_TikhonovSURE,
                    prof_pull_TikhonovSURE,
                    hist_coverage_TikhonovSURE,MAXTOY);
      }
      }
      //=========================================================
      // Tikhonov, minimum L-Curve curvature
      {
      if(!hist_toybase_TikhonovLCurve) {
         input=hist_unfoldingRecoFine_data;
      } else {
         input=generatePoissonToy(hist_toybase_TikhonovLCurve,itoy);
      }
      TUnfoldDensity tunfoldTikhonovLCurve
         (hist_migrationCoarseFine_MC,TUnfoldDensity::kHistMapOutputHoriz,
          TUnfoldDensity::kRegModeSize,TUnfoldDensity::kEConstraintNone,
          TUnfoldDensity::kDensityModeNone,binningCoarseGen,
          binningFineReco);
      tunfoldTikhonovLCurve.SubtractBackground(hist_unfoldingRecoFine_bgr,"bgr");
      tunfoldTikhonovLCurve.SetInput(input,biasScale);
      iBest=tunfoldTikhonovLCurve.ScanLcurve
         (NPOINT_TikhonovLCurve,0.,0.,
          (itoy==0) ? &graph_LCurve_TikhonovLCurve : 0,
          0,0,
          (itoy==0) ? &spline_Curvature_TikhonovLCurve : 0);
      if(itoy==0) {
         iBest_TikhonovLCurve=iBest;
         tauBest_TikhonovLCurve=tunfoldTikhonovLCurve.GetTau();
         DF_TikhonovLCurve=tunfoldTikhonovLCurve.GetDF();
         // all unfolded bins (here including fakes normalisation)
         hist_unfolded_TikhonovLCurve=
            tunfoldTikhonovLCurve.GetOutput("hist_unfolded_TikhonovLCurve",
             ";bin",0,0,false);
         hist_unfolded_TikhonovLCurve->Write();
         // unfolding result folded back -> for generating toys
         hist_toybase_TikhonovLCurve=tunfoldTikhonovLCurve.GetFoldedOutput
            ("hist_toybase_TikhonovLCurve",";bin",0,0,false,true);
         // unfolding result, signal only
         TH1 *hist_PTunfolded_TikhonovLCurve=
            tunfoldTikhonovLCurve.GetOutput("hist_PTunfolded_TikhonovLCurve",
                                     "P_{T,unfolded} [GeV]","signal");
         hist_PTunfolded_TikhonovLCurve->Write();
         // save auxillary plots
         graph_LCurve_TikhonovLCurve->Write("graph_LCurve_TikhonovLCurve");
         spline_Curvature_TikhonovLCurve->Write
            ("spline_Curvature_TikhonovLCurve");
      } else {
         TH1 *hist_unfolded_TikhonovLCurve_toy=
            tunfoldTikhonovLCurve.GetOutput
            (TString::Format("hist_PTunfolded_TikhonovLCurve_toy%d",itoy),
             ";bin",0,0,false);
         analyzeToy(hist_unfolded_TikhonovLCurve_toy,
                    hist_unfolded_TikhonovLCurve,
                    prof_pull_TikhonovLCurve,
                    hist_coverage_TikhonovLCurve,MAXTOY);

      }
      }
      //============================================================
      // Iterative method, SURE scan
      {
      if(itoy==0) {
         input=hist_unfoldingRecoFine_data;
      } else {
         input=generatePoissonToy(hist_toybase_IterativeSURE,itoy);
      }
      TUnfoldIterativeEM tunfoldIterative
         (hist_migrationCoarseFine_MC,TUnfoldDensity::kHistMapOutputHoriz,
          binningCoarseGen,binningFineReco);
      tunfoldIterative.SubtractBackground(hist_unfoldingRecoFine_bgr,"bgr");
      tunfoldIterative.SetInput(input,biasScale);
      iBest=tunfoldIterative.ScanSURE
         (NITER_Iterative,
          (itoy==0) ? &graph_SURE_IterativeSURE : 0,
          (itoy==0) ? &graph_DFdeviance_IterativeSURE :0);
      if(itoy==0) {
         iBest_IterativeSURE=iBest;
         // all unfolded bins (here including fakes normalisation)
         hist_unfolded_IterativeSURE=
            tunfoldIterative.GetOutput("hist_unfolded_IterativeSURE",
                                       ";bin",0,0,false);
         hist_unfolded_IterativeSURE->Write();
         // unfolding result folded back -> for generating toys
         hist_toybase_IterativeSURE=tunfoldIterative.GetFoldedOutput
            ("hist_toybase_IterativeSURE",";bin",0,0,false,true);
         // unfolding result, signal only
         TH1 *hist_PTunfolded_IterativeSURE=
            tunfoldIterative.GetOutput("hist_PTunfolded_IterativeSURE",
                                       "P_{T,unfolded} [GeV]","signal");
         hist_PTunfolded_IterativeSURE->Write();
         // save auxillary plots
         graph_SURE_IterativeSURE->Write("graph_SURE_IterativeSURE");
         graph_DFdeviance_IterativeSURE->Write
            ("graph_DFdeviance_IterativeSURE");
      } else {
         TH1 *hist_unfolded_IterativeSURE_toy=
            tunfoldIterative.GetOutput
            (TString::Format("hist_unfolded_IterativeSURE_toy%d",itoy),
             ";bin",0,0,false);
         analyzeToy(hist_unfolded_IterativeSURE_toy,
                    hist_unfolded_IterativeSURE,
                    prof_pull_IterativeSURE,
                    hist_coverage_IterativeSURE,MAXTOY);
      }
      }
      //============================================================
      // Iterative method, four iterations
      {
      if(itoy==0) {
         input=hist_unfoldingRecoFine_data;
      } else {
         input=generatePoissonToy(hist_toybase_IterativeFixed,itoy);
      }
      TUnfoldIterativeEM tunfoldIterative
         (hist_migrationCoarseFine_MC,TUnfoldDensity::kHistMapOutputHoriz,
          binningCoarseGen,binningFineReco);
      tunfoldIterative.SubtractBackground(hist_unfoldingRecoFine_bgr,"bgr");
      tunfoldIterative.SetInput(input,biasScale);
      tunfoldIterative.DoUnfold(NUM_FIXED_ITERATION);
      if(itoy==0) {
         // all unfolded bins (here including fakes normalisation)
         hist_unfolded_IterativeFixed=
            tunfoldIterative.GetOutput("hist_unfolded_IterativeFixed",
                                       ";bin",0,0,false);
         hist_unfolded_IterativeFixed->Write();
         // unfolding result folded back -> for generating toys
         hist_toybase_IterativeFixed=tunfoldIterative.GetFoldedOutput
            ("hist_toybase_IterativeFixed",";bin",0,0,false,true);
         // unfolding result, signal only
         TH1 *hist_PTunfolded_IterativeFixed=
            tunfoldIterative.GetOutput("hist_PTunfolded_IterativeFixed",
                                       "P_{T,unfolded} [GeV]","signal");
         hist_PTunfolded_IterativeFixed->Write();
      } else {
         TH1 *hist_unfolded_IterativeFixed_toy=
            tunfoldIterative.GetOutput
            (TString::Format("hist_unfolded_IterativeFixed_toy%d",itoy),
             ";bin",0,0,false);
         analyzeToy(hist_unfolded_IterativeFixed_toy,
                    hist_unfolded_IterativeFixed,
                    prof_pull_IterativeFixed,
                    hist_coverage_IterativeFixed,MAXTOY);
      }
      }
      
   }

   prof_pull_noRegularisation->Write();
   prof_pull_TikhonovSURE->Write();
   prof_pull_TikhonovLCurve->Write();
   prof_pull_IterativeSURE->Write();
   prof_pull_IterativeFixed->Write();

   hist_coverage_noRegularisation->Write();
   hist_coverage_TikhonovSURE->Write();
   hist_coverage_TikhonovLCurve->Write();
   hist_coverage_IterativeSURE->Write();
   hist_coverage_IterativeFixed->Write();

   // produce comparison plots
   // compare: (1) Tikhonov L-curve scan [L-curve]
   //          (2) Tikhonov SURE scan [SURE,DF,chi**2]
   //          (3) Iterative SURE scan [SURE,DF,deviance]

   TCanvas *canvas1=new TCanvas("compare","",900,300);
   canvas1->Divide(3,1);
   canvas1->cd(1);
   graph_LCurve_TikhonovLCurve->SetTitle
      (";log_{10}(#chi^{2}_{L});log_{10}(#chi^{2}_{A});");
   graph_LCurve_TikhonovLCurve->SetLineColor(kBlue);
   graph_LCurve_TikhonovLCurve->DrawClone("ALW");
   Double_t X_atMinCurvatue,Y_atMinCurvatue;
   graph_LCurve_TikhonovLCurve->GetPoint
      (iBest_TikhonovLCurve,X_atMinCurvatue,Y_atMinCurvatue);
   TGraph *minimum_LCurve=new TGraph(1,&X_atMinCurvatue,&Y_atMinCurvatue);
   minimum_LCurve->SetMarkerColor(kRed);
   minimum_LCurve->SetMarkerStyle(20);
   minimum_LCurve->SetMarkerSize(0.7);
   minimum_LCurve->DrawClone("P");
   TLegend *legend1=new TLegend(0.4,0.65,0.9,0.9,"Tikhonov, L-curve");
   legend1->SetBorderSize(0);
   legend1->SetFillStyle(0);
   legend1->SetTextSize(0.045);
   legend1->AddEntry(graph_LCurve_TikhonovLCurve,"L-curve","l");
   
   legend1->AddEntry(minimum_LCurve,"largest curvature","p");
   legend1->AddEntry
      ((TObject *)0,TString::Format("at #tau=%.3g",tauBest_TikhonovLCurve),"");
   legend1->AddEntry((TObject *)0,TString::Format("D.F.=%3g",DF_TikhonovLCurve),"");
   legend1->Draw();
   canvas1->cd(2);
   gPad->SetLogy();
   double yMin=1.0;
   double yLine=10.;
   double yMax=300.;
   graph_logTauSURE_TikhonovSURE->GetYaxis()->SetRangeUser(yMin,yMax);
   graph_logTauSURE_TikhonovSURE->GetXaxis()->SetRangeUser(-5.,-1.);
   graph_logTauSURE_TikhonovSURE->SetTitle(";log_{10}(#tau)");
   graph_logTauSURE_TikhonovSURE->SetLineColor(kBlue);
   graph_logTauSURE_TikhonovSURE->DrawClone();
   int n_scanSURE=graph_logTauSURE_TikhonovSURE->GetN();
   double const *logTau_scanSURE=graph_logTauSURE_TikhonovSURE->GetX();
   double const *DF_scanSURE=graph_dfChi2A_TikhonovSURE->GetX();
   double const *chi2A_scanSURE=graph_dfChi2A_TikhonovSURE->GetY();


   TGraph *logTauDF=new TGraph(n_scanSURE,logTau_scanSURE,DF_scanSURE);
   TGraph *logTauChi2A=new TGraph(n_scanSURE,logTau_scanSURE,chi2A_scanSURE);
   logTauDF->SetLineColor(kRed);
   logTauDF->DrawClone("L");
   logTauChi2A->SetLineColor(kMagenta);
   logTauChi2A->DrawClone("L");
   double tikhonov_logTauSURE=logTau_scanSURE[iBest_TikhonovSURE];
   TLine *line=new TLine(tikhonov_logTauSURE,yLine,
                         tikhonov_logTauSURE,yMax);
   line->SetLineStyle(2);
   line->Draw();
   TLegend *legend2=new TLegend(0.25,0.2,0.9,0.45,"Tikhonov, ,minimize SURE");
   legend2->SetBorderSize(0);
   legend2->SetFillStyle(0);
   legend2->SetTextSize(0.045);
   legend2->AddEntry(graph_logTauSURE_TikhonovSURE,"SURE","l");
   legend2->AddEntry(logTauDF,"D.F.","l");
   legend2->AddEntry(logTauChi2A,"#chi^{2}_{A}","l");
   legend2->AddEntry(line,TString::Format
                     ("min(SURE) at #tau=%.3g",TMath::Power(10.,tikhonov_logTauSURE)),"l");
   legend2->AddEntry((TObject *)0,TString::Format
                     ("D.F.=%3g",DF_scanSURE[iBest_TikhonovSURE]),"");
   legend2->Draw();
   canvas1->cd(3);
   gPad->SetLogy();
   graph_SURE_IterativeSURE->GetYaxis()->SetRangeUser(yMin,yMax);
   graph_SURE_IterativeSURE->GetXaxis()->SetRangeUser(-1.5,100.5);
   graph_SURE_IterativeSURE->SetTitle(";iteration");
   graph_SURE_IterativeSURE->SetMarkerColor(kBlue);
   graph_SURE_IterativeSURE->SetMarkerStyle(20);
   graph_SURE_IterativeSURE->SetMarkerSize(0.3);
   graph_SURE_IterativeSURE->DrawClone("APW");
   int n_scanSURE_iterative=graph_SURE_IterativeSURE->GetN();
   double const *nIter_scanSURE_iterative=graph_SURE_IterativeSURE->GetX();
   double const *DF_scanSURE_iterative=graph_DFdeviance_IterativeSURE->GetX();
   double const *deviance_scanSURE=graph_DFdeviance_IterativeSURE->GetY();
   TGraph *DF_iterative=new TGraph
      (n_scanSURE_iterative,nIter_scanSURE_iterative,DF_scanSURE_iterative);
   TGraph *deviance_iterative=new TGraph
      (n_scanSURE_iterative,nIter_scanSURE_iterative,deviance_scanSURE);
   DF_iterative->SetMarkerColor(kRed);
   DF_iterative->SetMarkerStyle(24);
   DF_iterative->SetMarkerSize(0.3);
   DF_iterative->DrawClone("P");
   deviance_iterative->SetMarkerColor(kMagenta);
   deviance_iterative->SetMarkerStyle(22);
   deviance_iterative->SetMarkerSize(0.3);
   deviance_iterative->DrawClone("P");
   TLine *line2=new TLine(iBest_IterativeSURE,yLine,iBest_IterativeSURE,yMax);
   line2->SetLineStyle(2);
   line2->Draw();
   TLegend *legend3=new TLegend(0.25,0.2,0.9,0.45,"Iterative EM, minimize SURE");
   legend3->SetBorderSize(0);
   legend3->SetFillStyle(0);
   legend3->SetTextSize(0.045);
   legend3->AddEntry(graph_SURE_IterativeSURE,"SURE","p");
   legend3->AddEntry(DF_iterative,"D.F.","p");
   legend3->AddEntry(deviance_iterative,"deviance","p");
   
   legend3->AddEntry(line2,TString::Format
                     ("min(SURE) at iteration=%d",iBest_IterativeSURE),"l");
   legend3->AddEntry((TObject *)0,TString::Format
                     ("D.F.=%3g",DF_scanSURE_iterative[iBest_IterativeSURE]),"");
   legend3->Draw();

   canvas1->SaveAs("testunfold2c_scan.eps");

   TCanvas *canvas2=new TCanvas("coverage","",900,300);
   canvas2->Divide(3,1);

   hist_coverage_noRegularisation->SetLineColor(kBlue);
   hist_coverage_noRegularisation->SetMarkerStyle(0);
   canvas2->cd(1);

   hist_coverage_TikhonovSURE->GetYaxis()->SetRangeUser(0.,1.);
   hist_coverage_TikhonovSURE->SetTitle(";bin number;coverage");
   hist_coverage_TikhonovSURE->SetFillStyle(1001);
   hist_coverage_TikhonovSURE->SetFillColor(kCyan-10);
   hist_coverage_TikhonovSURE->SetLineColor(kCyan);
   hist_coverage_TikhonovSURE->DrawCopy("HIST");
   hist_coverage_noRegularisation->DrawCopy("SAME HIST");

   TLegend *legend21=new TLegend(0.2,0.78,0.6,0.93);
   legend21->SetFillStyle(0);
   legend21->SetBorderSize(0);
   legend21->SetTextSize(0.06);

   legend21->AddEntry(hist_coverage_noRegularisation,"no regularisation","l");
   legend21->AddEntry(hist_coverage_TikhonovSURE,"Tikhonov + SURE","lf");
   legend21->Draw();

   canvas2->cd(2);
   
   hist_coverage_IterativeSURE->GetYaxis()->SetRangeUser(0.,1.);
   hist_coverage_IterativeSURE->SetTitle(";bin number;coverage");
   hist_coverage_IterativeSURE->SetFillStyle(1001);
   hist_coverage_IterativeSURE->SetFillColor(kRed-10);
   hist_coverage_IterativeSURE->SetLineColor(kRed);
   hist_coverage_IterativeSURE->DrawCopy("HIST");
   hist_coverage_noRegularisation->DrawCopy("SAME HIST");

   TLegend *legend22=new TLegend(0.2,0.78,0.6,0.93);
   legend22->SetFillStyle(0);
   legend22->SetBorderSize(0);
   legend22->SetTextSize(0.06);
   legend22->AddEntry(hist_coverage_noRegularisation,"no regularisation","l");
   legend22->AddEntry(hist_coverage_IterativeSURE,"Iterative + SURE","lf");
   legend22->Draw();

   canvas2->cd(3);

   hist_coverage_IterativeFixed->GetYaxis()->SetRangeUser(0.,1.);
   hist_coverage_IterativeFixed->SetTitle(";bin number;coverage");
   hist_coverage_IterativeFixed->SetFillStyle(1001);
   hist_coverage_IterativeFixed->SetFillColor(kGray);
   hist_coverage_IterativeFixed->SetLineColor(kGray+3);
   hist_coverage_IterativeFixed->DrawCopy("HIST");
   hist_coverage_noRegularisation->DrawCopy("SAME HIST");

   TLegend *legend23=new TLegend(0.2,0.78,0.6,0.93);
   legend23->SetFillStyle(0);
   legend23->SetBorderSize(0);
   legend23->SetTextSize(0.06);
   legend23->AddEntry(hist_coverage_noRegularisation,"no regularisation","l");
   legend23->AddEntry(hist_coverage_IterativeFixed,"Iterative, N=4","lf");
   legend23->Draw();



   canvas2->SaveAs("testunfold2c_coverageFromData.eps");
   
   delete outputFile;

}

static TH1 *generatePoissonToy(TH1 *base,int ntoy) {
   static TRandom *rnd=0;
   if(!rnd) rnd=new TRandom3();
   TH1 *r=(TH1 *)base->Clone(base->GetName()+TString::Format("_toy%d",ntoy));
   for(int ibin=0;ibin<=r->GetNbinsX()+1;ibin++) {
      double mu=r->GetBinContent(ibin);
      double c=0.;
      if(mu>0.) {
         c=rnd->Poisson(mu);
      }
      r->SetBinContent(ibin,c);
      r->SetBinError(ibin,TMath::Sqrt(c));
   }
   return r;
}

static void analyzeToy(TH1 const *hist_toy,
                       TH1 const *hist_truth,
                       TProfile *&prof_pull,TH1 *&hist_coverage,int nToyTotal) {
   if(!prof_pull) {
      TString namePull(hist_truth->GetName()+TString("_toyPull"));
      TString nameCoverage(hist_truth->GetName()+TString("_toyCoverage"));
      TString title=hist_truth->GetTitle();
      TArrayD const *xBins=hist_toy->GetXaxis()->GetXbins();
      if(xBins && (xBins->GetSize()>1)) {
         prof_pull=new TProfile
            (namePull,title,xBins->GetSize()-1,xBins->GetArray());
         hist_coverage=new TH1D
            (nameCoverage,title,xBins->GetSize()-1,xBins->GetArray());
      } else {
         int nBins=hist_toy->GetNbinsX();
         double x0=hist_toy->GetXaxis()->GetXmin();
         double x1=hist_toy->GetXaxis()->GetXmax();
         prof_pull=new TProfile(namePull,title,nBins,x0,x1);
         hist_coverage=new TH1D(nameCoverage,title,nBins,x0,x1);
      }
   }
   
   for(int i=0;i<=hist_toy->GetNbinsX()+1;i++) {
      double pullI=(hist_toy->GetBinContent(i)-hist_truth->GetBinContent(i))/
         hist_truth->GetBinError(i);
      prof_pull->Fill
         (hist_toy->GetBinCenter(i),pullI);
      if(TMath::Abs(pullI)<1.) {
         hist_coverage->Fill(hist_toy->GetBinCenter(i),1./nToyTotal);
      }
   }
}
