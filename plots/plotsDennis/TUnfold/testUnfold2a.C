// Author: Stefan Schmitt
// DESY, July 2016

//  Version 17.9, example of using the SURE method

#include <iostream>
#include <map>
#include <cmath>
#include <TMath.h>
#include <TRandom3.h>
#include <TFile.h>
#include <TTree.h>
#include <TLorentzVector.h>

#define MASS1 0.511E-3

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

TRandom *g_rnd=0;

class ToyEvent7 {
public:
   void GenerateDataEvent(TRandom *rnd);
   void GenerateSignalEvent(TRandom *rnd);
   void GenerateBgrEvent(TRandom *rnd);
   // reconstructed quantities
   inline Double_t GetMRec(int i) const { return fMRec[i]; }
   inline Double_t GetPtRec(int i) const { return fPtRec[i]; }
   inline Double_t GetEtaRec(int i) const { return fEtaRec[i]; }
   inline Double_t GetPhiRec(int i) const { return fPhiRec[i]; }
   inline Int_t HasTriggered(void) const { return fHasTriggered; }

   // generator level quantities
   inline Double_t GetMGen(int i) const { 
      if(IsSignal()) return fMGen[i];
      else return -1.0;
   }
   inline Double_t GetPtGen(int i) const { 
      if(IsSignal()) return fPtGen[i];
      else return -1.0;
   }
   inline Double_t GetEtaGen(int i) const {
       if(IsSignal()) return fEtaGen[i];
       else return 999.0;
   }
   inline Double_t GetPhiGen(int i) const {
       if(IsSignal()) return fPhiGen[i];
       else return 999.0;
   }
   inline Bool_t IsSignal(void) const { return fIsSignal; }
protected:

   void GenerateSignalKinematics(TRandom *rnd,Bool_t isData);
   void GenerateBgrKinematics(TRandom *rnd,Bool_t isData);
   void GenerateReco(TRandom *rnd);

   // reconstructed quantities
   Double_t fMRec[3];
   Double_t fPtRec[3];
   Double_t fEtaRec[3];
   Double_t fPhiRec[3];
   Int_t fHasTriggered;
   // generated quantities
   Double_t fMGen[3];
   Double_t fPtGen[3];
   Double_t fEtaGen[3];
   Double_t fPhiGen[3];
   Bool_t fIsSignal;
public:
   static Double_t kDataSignalFraction;
   static Double_t kMCSignalFraction;

};

void testUnfold2a()
{
  // random generator
  g_rnd=new TRandom3(4711);

  // data and MC number of events
  Double_t muData0=200000.;
  // luminosity error
  Double_t muData=muData0*g_rnd->Gaus(1.0,0.03);
  // stat error
  Int_t neventData      = g_rnd->Poisson( muData);

  // generated number of MC events
  Int_t neventSigmc     = 1000000;
  Int_t neventBgrmc     = 1000000;

  Float_t etaRec[3],ptRec[3],phiRec[3],mRec[3];
  Float_t etaGen[3],ptGen[3],phiGen[3],mGen[3];
  Float_t weight;
  Int_t hasTriggered,isSignal;

  //==================================================================  
  // Step 1: generate data TTree

  TFile *dataFile=new TFile("testUnfold2a_data.root","recreate");
  TTree *dataTree=new TTree("event","event");
  TTree *dataTruthTree=new TTree("DataTruth","event");
  TTree *dataBgrTree=new TTree("DataBgr","event");

  dataTree->Branch("etaRec",etaRec+2,"etaRec/F");
  dataTree->Branch("ptRec",ptRec+2,"ptRec/F");
  dataTree->Branch("phiRec",phiRec+2,"phiRec/F");
  dataTree->Branch("mRec",mRec+2,"mRec/F");
  dataTree->Branch("etaRec2",etaRec,"etaRec2[2]/F");
  dataTree->Branch("ptRec2",ptRec,"ptRec2[2]/F");
  dataTree->Branch("phiRec2",phiRec,"phiRec2[2]/F");
  dataTree->Branch("hasTriggered",&hasTriggered,"hasTriggered/I");
  dataTree->Branch("weightRec",&weight,"weightRec/F");

  // for real data, only the triggered events are available
  // in this study, the data truth parameters are stored in secondary TTrees
  dataBgrTree->Branch("etaRec",etaRec+2,"etaRec/F");
  dataBgrTree->Branch("ptRec",ptRec+2,"ptRec/F");
  dataBgrTree->Branch("phiRec",phiRec+2,"phiRec/F");
  dataBgrTree->Branch("mRec",mRec+2,"mRec/F");
  dataBgrTree->Branch("etaRec2",etaRec,"etaRec2[2]/F");
  dataBgrTree->Branch("ptRec2",ptRec,"ptRec2[2]/F");
  dataBgrTree->Branch("phiRec2",phiRec,"phiRec2[2]/F");
  dataBgrTree->Branch("hasTriggered",&hasTriggered,"hasTriggered/I");
  dataBgrTree->Branch("weightRec",&weight,"weightRec/F");

  dataTruthTree->Branch("etaGen",etaGen+2,"etaGen/F");
  dataTruthTree->Branch("ptGen",ptGen+2,"ptGen/F");
  dataTruthTree->Branch("phiGen",phiGen+2,"phiGen/F");
  dataTruthTree->Branch("etaGen2",etaGen,"etaGen2[2]/F");
  dataTruthTree->Branch("ptGen2",ptGen,"ptGen2[2]/F");
  dataTruthTree->Branch("phiGen2",phiGen,"phiGen2[2]/F");
  dataTruthTree->Branch("mGen",mGen+2,"mGen/F");
  dataTruthTree->Branch("weightGen",&weight,"weightGen/F");

  cout<<"fill data tree\n";

  weight=1.;

  //Int_t nEvent=0,nTriggered=0;
  for(int ievent=0;ievent<neventData;ievent++) {
     ToyEvent7 event;
     event.GenerateDataEvent(g_rnd);
     for(int i=0;i<3;i++) {
        etaRec[i]=event.GetEtaRec(i);
        ptRec[i]=event.GetPtRec(i);
        phiRec[i]=event.GetPhiRec(i);
        mRec[i]=event.GetMRec(i);
        etaGen[i]=event.GetEtaGen(i);
        ptGen[i]=event.GetPtGen(i);
        phiGen[i]=event.GetPhiGen(i);
        mGen[i]=event.GetMGen(i);
     }
     hasTriggered=event.HasTriggered();
     if(hasTriggered) {
        dataTree->Fill();
        if(event.IsSignal()) {
           dataBgrTree->Fill();
        }
     }
     if(event.IsSignal()) {
        dataTruthTree->Fill();
     }
     if(!(ievent%100000)) cout<<"   data event "<<ievent<<"\n";
  }

  dataTree->Write();
  dataTruthTree->Write();
  delete dataTree;
  delete dataTruthTree;
  delete dataBgrTree;
  delete dataFile;

  //==================================================================  
  // Step 2: Generate signal TTree

  TFile *mcFile=new TFile("testUnfold2a_MC.root","recreate");
  TTree *mcTree=new TTree("event","event");

  mcTree->Branch("etaRec",etaRec+2,"etaRec/F");
  mcTree->Branch("ptRec",ptRec+2,"ptRec/F");
  mcTree->Branch("phiRec",ptRec+2,"phiRec/F");
  mcTree->Branch("mRec",mRec+2,"mRec/F");
  mcTree->Branch("etaRec2",etaRec,"etaRec2[2]/F");
  mcTree->Branch("ptRec2",ptRec,"ptRec2[2]/F");
  mcTree->Branch("phiRec2",phiRec,"phiRec2[2]/F");
  mcTree->Branch("hasTriggered",&hasTriggered,"hasTriggered/I");
  mcTree->Branch("weightRec",&weight,"weightRec/F");
  mcTree->Branch("etaGen",etaGen+2,"etaGen/F");
  mcTree->Branch("ptGen",ptGen+2,"ptGen/F");
  mcTree->Branch("phiGen",phiGen+2,"phiGen/F");
  mcTree->Branch("mGen",mGen+2,"mGen/F");
  mcTree->Branch("etaGen2",etaGen,"etaGen2[2]/F");
  mcTree->Branch("ptGen2",ptGen,"ptGen2[2]/F");
  mcTree->Branch("phiGen2",phiGen,"phiGen2[2]/F");
  mcTree->Branch("weightGen",&weight,"weightGen/F");
  mcTree->Branch("isSignal",&isSignal,"isSignal/I");

  cout<<"fill MC tree\n";
  cout<<" (signal events)\n";

  weight=ToyEvent7::kMCSignalFraction*muData0/neventSigmc;
 
  isSignal=1;
 
  for(int ievent=0;ievent<neventSigmc;ievent++) {
     ToyEvent7 event;
     event.GenerateSignalEvent(g_rnd);

     for(int i=0;i<3;i++) {
        etaRec[i]=event.GetEtaRec(i);
        ptRec[i]=event.GetPtRec(i);
        phiRec[i]=event.GetPhiRec(i);
        mRec[i]=event.GetMRec(i);
        etaGen[i]=event.GetEtaGen(i);
        ptGen[i]=event.GetPtGen(i);
        phiGen[i]=event.GetPhiGen(i);
        mGen[i]=event.GetMGen(i);
     }
     hasTriggered=event.HasTriggered();

     if(!(ievent%100000)) cout<<"   signal event "<<ievent<<"\n";

     mcTree->Fill();
  }

  // ==============================================================
  // Step 3: Generate background MC TTree

  cout<<" (background events)\n";

  weight=(1.-ToyEvent7::kMCSignalFraction)*muData0/neventBgrmc;

  isSignal=0;
  for(int i=0;i<3;i++) {
     etaGen[i]=0.;
     ptGen[i]=0.;
     phiGen[i]=0.;
     mGen[i]=0.;
  }

  for(int ievent=0;ievent<neventBgrmc;ievent++) {
     ToyEvent7 event;
     event.GenerateBgrEvent(g_rnd);
     for(int i=0;i<3;i++) {
        etaRec[i]=event.GetEtaRec(i);
        ptRec[i]=event.GetPtRec(i);
        phiRec[i]=event.GetPhiRec(i);
        mRec[i]=event.GetMRec(i);
     }
     hasTriggered=event.HasTriggered() ? 1 : 0;

     if(!(ievent%100000)) cout<<"   background event "<<ievent<<"\n";

     if(hasTriggered) {
        mcTree->Fill();
     }
  }

  mcTree->Write();
  delete mcTree;
  delete mcFile;

}

Double_t ToyEvent7::kDataSignalFraction=0.18;
Double_t ToyEvent7::kMCSignalFraction=0.22;

void ToyEvent7::GenerateDataEvent(TRandom *rnd) {
   fIsSignal=rnd->Uniform()<kDataSignalFraction;
   if(IsSignal()) {
      GenerateSignalKinematics(rnd,kTRUE);
   } else {
      GenerateBgrKinematics(rnd,kTRUE);
   }
   GenerateReco(rnd);
}

void ToyEvent7::GenerateSignalEvent(TRandom *rnd) {
   fIsSignal=1;
   GenerateSignalKinematics(rnd,kFALSE);
   GenerateReco(rnd);
}

void ToyEvent7::GenerateBgrEvent(TRandom *rnd) {
   fIsSignal=0;
   GenerateBgrKinematics(rnd,kFALSE);
   GenerateReco(rnd);
}

void ToyEvent7::GenerateSignalKinematics(TRandom *rnd,Bool_t isData) {

   // fake decay of Z0 to two fermions
   double M0=91.1876;
   double Gamma=2.4952;
   // generated mass
   do {
      fMGen[2]=rnd->BreitWigner(M0,Gamma);
   } while(fMGen[2]<=0.0);

   double MAX_ETA=9.0;
   double POW_ETA=0.0;
   double MU_PT=5.;
   double MU_PTETA=0.;
   double SIGMA_PT=2.2;
   double SIGMA_PTETA=0.0;
   double DECAY_A=0.2;
   if(isData) {
      POW_ETA=0.1;
      MU_PT=6.;
      SIGMA_PT=1.8;
      MU_PTETA=0.0;
      SIGMA_PTETA=0.05;
      DECAY_A=0.3;
   }
   double eta=rnd->Uniform(-MAX_ETA,MAX_ETA);
   if(eta!=0.0) {
      eta *= TMath::Power((TMath::Abs(eta)/MAX_ETA),POW_ETA);
   }
   fEtaGen[2]=eta;
   fPhiGen[2]=rnd->Uniform(-M_PI,M_PI);
   do {
      fPtGen[2]=rnd->Landau(MU_PT+TMath::Exp(MU_PTETA*eta*eta),
                            SIGMA_PT+SIGMA_PTETA*eta*eta);
   } while((fPtGen[2]<=0.0)||(fPtGen[2]>500.));
   //========================== decay
   TLorentzVector sum;
   sum.SetPtEtaPhiM(fPtGen[2],fEtaGen[2],fPhiGen[2],fMGen[2]);
   // boost into lab-frame
   TVector3 boost=sum.BoostVector();
   // decay in rest-frame

   TLorentzVector p[3];
   double m=MASS1;
   double costh;
   do {
      double r=rnd->Uniform(-1.,1.);
      costh=r*(1.+DECAY_A*r*r);
   } while(fabs(costh)>=1.0);
   double phi=rnd->Uniform(-M_PI,M_PI);
   double e=0.5*sum.M();
   double ptot=TMath::Sqrt(e+m)*TMath::Sqrt(e-m);
   double pz=ptot*costh;
   double pt=TMath::Sqrt(ptot+pz)*TMath::Sqrt(ptot-pz);
   double px=pt*cos(phi);
   double py=pt*sin(phi);
   p[0].SetXYZT(px,py,pz,e);
   p[1].SetXYZT(-px,-py,-pz,e);
   for(int i=0;i<2;i++) {
      p[i].Boost(boost);
   }
   p[2]=p[0]+p[1];
   for(int i=0;i<3;i++) {
      fPtGen[i]=p[i].Pt();
      fEtaGen[i]=p[i].Eta();
      fPhiGen[i]=p[i].Phi();
      fMGen[i]=p[i].M();
   }
}

void ToyEvent7::GenerateBgrKinematics(TRandom *rnd,Bool_t isData) {
   for(int i=0;i<3;i++) {
      fPtGen[i]=0.0;
      fEtaGen[i]=0.0;
      fPhiGen[i]=0.0;
   }
   TLorentzVector p[3];
   for(int i=0;i<2;i++) {
      p[i].SetPtEtaPhiM(rnd->Exp(15.0),rnd->Uniform(-5.,5.),
                        rnd->Uniform(-M_PI,M_PI),isData ? MASS1 : MASS1);
   }
   p[2]=p[0]+p[1];
   for(int i=0;i<3;i++) {
      fPtRec[i]=p[i].Pt();
      fEtaRec[i]=p[i].Eta();
      fPhiRec[i]=p[i].Phi();
      fMRec[i]=p[i].M();
   }
}

void ToyEvent7::GenerateReco(TRandom *rnd) {
   if(fIsSignal) {
      TLorentzVector p[3];
      for(int i=0;i<2;i++) {
         Double_t expEta=TMath::Exp(fEtaGen[i]);
         Double_t coshEta=(expEta+1./expEta);
         Double_t eGen=fPtGen[i]*coshEta;
         Double_t sigmaE=
            0.1*TMath::Sqrt(eGen)
            +1.0*coshEta
            +0.01*eGen;
         Double_t eRec;
         do {
            eRec=rnd->Gaus(eGen,sigmaE);
         } while(eRec<=0.0);
         Double_t sigmaEta=0.1+0.05*TMath::Abs(fEtaGen[i]);
         p[i].SetPtEtaPhiM(eRec/(expEta+1./expEta),
                           rnd->Gaus(fEtaGen[i],sigmaEta),
                           remainder(rnd->Gaus(fPhiGen[i],0.03),2.*M_PI),
                           MASS1);
      }
      p[2]=p[0]+p[1];
      for(int i=0;i<3;i++) {
         fPtRec[i]=p[i].Pt();
         fEtaRec[i]=p[i].Eta();
         fPhiRec[i]=p[i].Phi();
         fMRec[i]=p[i].M();
      }
   }
   // simulate acceptance
   bool isReconstructed=true;
   for(int i=0;i<2;i++) {
      if((rnd->Uniform()>0.98/(TMath::Exp(-(fPtRec[i]-10.5)/0.5)+1.))||
         (rnd->Uniform()>1./(TMath::Exp(-(4.5-TMath::Abs(fEtaRec[i]))/0.5)+1))) {
         isReconstructed=false;
         fPtRec[i]=0.;
         fEtaRec[i]=0.;
         fPhiRec[i]=0.;
         fMRec[i]=0.;
         fPtRec[2]=0.;
         fEtaRec[2]=0.;
         fPhiRec[2]=0.;
         fMRec[2]=0.;
      }
   }

   // trigger on a single lepton above threshold and two reconstructed leptons
   fHasTriggered=0;
   if(isReconstructed)
      for(int i=0;i<2;i++) {
         if((rnd->Uniform()<0.92/(TMath::Exp(-(fPtRec[i]-15.5)/0.5)+1.))&&
            (rnd->Uniform()<1./(TMath::Exp(-(2.5-TMath::Abs(fEtaRec[i]))/0.1)+1))) {
            fHasTriggered |= (i+1);
      }
   }
}
