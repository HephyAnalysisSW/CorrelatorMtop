#include "unfolding.h"


unfolding::unfolding(TH1F* input, TH2F* matrix, TUnfoldBinning* recBinning, TUnfoldBinning* genBinning){
  unfolding_done = false;
  h_input = input;
  h_response_matrix = matrix;
  binning_gen = genBinning;
  binning_rec = recBinning;

  // ---------------------------------------------------------------------------
  // setup unfolding
  histMode = TUnfold::kHistMapOutputHoriz;

  constraintMode = TUnfold::kEConstraintNone;

  // regMode = TUnfold::kRegModeSize;
  // regMode = TUnfold::kRegModeDerivative;
  regMode = TUnfold::kRegModeCurvature;

  densityFlags = TUnfoldDensity::kDensityModeNone;
  // densityFlags = TUnfoldDensity::kDensityModeBinWidth;
  // densityFlags = TUnfoldDensity::kDensityModeBinWidthAndUser;
  // densityFlags = TUnfoldDensity::kDensityModeUser;

  // scanMode = TUnfoldDensity::kEScanTauRhoAvg;
  // scanMode = TUnfoldDensity::kEScanTauRhoMax;
  scanMode = TUnfoldDensity::kEScanTauRhoAvgSys;
  // scanMode = TUnfoldDensity::kEScanTauRhoSquareAvg;
  // scanMode = TUnfoldDensity::kEScanTauRhoSquareAvgSys;

  REGULARISATION_DISTRIBUTION=0;
  REGULARISATION_AXISSTEERING=0;
  SCAN_DISTRIBUTION=0;
  SCAN_AXISSTEERING=0;

  cout << "   -->  Regularisation Mode: ";
  if(regMode == TUnfold::kRegModeSize) cout<< "Size";
  else if(regMode == TUnfold::kRegModeDerivative) cout<< "Derivative";
  else if(regMode == TUnfold::kRegModeCurvature) cout<< "Curvature";
  cout << endl;

  return;
}


void unfolding::unfold(){
  // set up TUnfold Class
  TUnfoldDensity unfolder(
    h_response_matrix,
    histMode,
    regMode,
    constraintMode,
    densityFlags,
    binning_gen,
    binning_rec,
    REGULARISATION_DISTRIBUTION,
    REGULARISATION_AXISSTEERING
  );


  unfolder.SetInput(h_input, scale_fb0);
  // unfolder.SubtractBackground(backgrounds[i], bgr_name[i], 1.0, scale_error);

  TSpline *logTauX=0,*logTauY=0;
  double tau_min = 0.00001;
  double tau_max = 0.9;
  int nscan = 100;

  if(uMode == unfoldingMode::LScan){
    cout << "Unfolding with L-curve scan and " << nscan << " points" << endl;
    unfolder.ScanLcurve(nscan,tau_min,tau_max,&lcurve,&logTauX,&logTauY);
    // get tau value and position on l-curve
    tau = unfolder.GetTau();
    double logTau = TMath::Log10(tau);
    lcurveX = logTauX->Eval(logTau);
    lcurveY = logTauY->Eval(logTau);
  }
  else if(uMode == unfoldingMode::FixedTau){
    cout << "Unfolding with fixed tau = " << tau_fix << endl;
    unfolder.DoUnfold(tau_fix);
  }
  else if(uMode == unfoldingMode::RhoScan){
    cout << "Unfolding with Rho scan" << endl;
    unfolder.ScanTau(nscan,tau_min,tau_max,&rhoLogTau, scanMode, SCAN_DISTRIBUTION, SCAN_AXISSTEERING, &lcurve,&logTauX,&logTauY);
    // get tau value and position on l-curve
    tau=unfolder.GetTau();
    double logTau=TMath::Log10(tau);
    lcurveX=logTauX->Eval(logTau);
    lcurveY=logTauY->Eval(logTau);
  }

  // axis steering for outputs
  h_unfolded = (TH1F*) unfolder.GetOutput("unfolded", 0,"measurement_gen", 0, kFALSE); // returns a 1D hist with bin numbers
  h_unfolded_2D = (TH2F*) unfolder.GetOutput("unfolded_2D", 0,"measurement_gen", 0, kTRUE); // returns a 2D hist
  unfolding_done = true;
}

void unfolding::set_unfolding_mode(unfoldingMode mode){
  uMode = mode;
}

void unfolding::set_fixed_tau(double t){
  tau_fix = t;
}

void unfolding::set_offset_scale(double scale){
  cout << "Setting bias scale to " << scale << endl;
  scale_fb0 = scale;
}

TH1F* unfolding::get_unfolded(){
  if(!unfolding_done) cout << "Unfolding has not been done! Return empty histogram." << endl;
  return h_unfolded;
}

TH2F* unfolding::get_unfolded_2D(){
  if(!unfolding_done) cout << "Unfolding has not been done! Return empty histogram." << endl;
  return h_unfolded_2D;
}
