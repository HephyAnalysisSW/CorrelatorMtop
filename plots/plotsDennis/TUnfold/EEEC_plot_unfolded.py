#!/usr/bin/env python

import ROOT
import array
import os
import Analysis.Tools.syncer

from math                           import sqrt
from CorrelatorMtop.Tools.helpers   import getObjFromFile
from MyRootTools.plotter.Plotter    import Plotter

ROOT.gROOT.SetBatch(ROOT.kTRUE)

def project2Dto1D(hist, log=False):
    Nbins_x = hist.GetXaxis().GetNbins()
    Nbins_y = hist.GetYaxis().GetNbins()
    zeta_thresholds = []
    contents = [0]*Nbins_x
    errors2 = [0]*Nbins_x
    for i in range(Nbins_x):
        bin_x = i+1
        if bin_x==1:
            zeta_thresholds.append(hist.GetXaxis().GetBinLowEdge(bin_x))
        zeta_thresholds.append(hist.GetXaxis().GetBinUpEdge(bin_x))
        for j in range(Nbins_y):
            bin_y = j+1
            if log:
                # if bins are log, get the geometric mean
                low_thresh = hist.GetYaxis().GetBinLowEdge(bin_y)
                high_thresh = hist.GetYaxis().GetBinUpEdge(bin_y)
                weight = sqrt(low_thresh*high_thresh)
            else:
                weight = hist.GetYaxis().GetBinCenter(bin_y)

            content = hist.GetBinContent(bin_x, bin_y)
            error = hist.GetBinError(bin_x, bin_y)
            # print(bin_x, bin_y, content, weight)
            contents[i] += content*weight
            errors2[i] += pow(error*weight, 2)
    # create 1D hist
    projection = ROOT.TH1F("projection", "", len(zeta_thresholds)-1, array.array('d', zeta_thresholds))
    for i in range(len(zeta_thresholds)-1):
        bin = i+1
        projection.SetBinContent(bin, contents[i])
        projection.SetBinError(bin, sqrt(errors2[i]))
    return projection

def makeSameBinning(hist, dummy):
    newHist = dummy.Clone(hist.GetName()+"_newBinning")
    Nbins_x = dummy.GetXaxis().GetNbins()
    for i in range(Nbins_x):
        bin = i+1
        newHist.SetBinContent(bin, hist.GetBinContent(bin))
        newHist.SetBinError(bin, hist.GetBinError(bin))
    return newHist

# truth distributions
h_truth = getObjFromFile("EEEC_Histograms.root", "gen__pseudodata")
h_truth_weighted_oldBinning = getObjFromFile("EEEC_Histograms.root", "gen_weighted__pseudodata")
h_truth_weighted_2D = getObjFromFile("EEEC_Result.root", "unfolded_2D_self_pseudodata")
h_truth_weighted_proj = project2Dto1D(h_truth_weighted_2D, log=True)

# bias distributions
h_bias = getObjFromFile("EEEC_Histograms.root", "gen__ttbar")
h_bias_weighted_oldBinning = getObjFromFile("EEEC_Histograms.root", "gen_weighted__ttbar")
h_bias_weighted_2D = getObjFromFile("EEEC_Result.root", "unfolded_2D_self_ttbar")
h_bias_weighted_proj = project2Dto1D(h_bias_weighted_2D, log=True)

# unfolded
h_unfolded = getObjFromFile("EEEC_Result.root", "unfolded")
h_unfolded_2D = getObjFromFile("EEEC_Result.root", "unfolded_2D")
h_unfolded_proj = project2Dto1D(h_unfolded_2D, log=True)
h_truth_weighted = makeSameBinning(h_truth_weighted_oldBinning, h_unfolded_proj)
h_bias_weighted = makeSameBinning(h_truth_weighted_oldBinning, h_unfolded_proj)

p = Plotter("Unfolded")
p.plot_dir = "/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/plots/Unfolding"
p.lumi = "60"
p.xtitle = "Unfolding bin"
p.ytitle = "Events"
p.NcolumnsLegend = 1
p.drawRatio = True
p.addBackground(h_truth, "Particle level", ROOT.kAzure+7)
p.addSignal(h_bias, "Bias distribution", ROOT.kRed-2)
p.addData(h_unfolded, "Unfolded")
p.draw()


p = Plotter("Unfolded_proj")
p.plot_dir = "/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/plots/Unfolding"
p.lumi = "60"
p.xtitle = "#it{p}_{T} weighted #zeta"
p.ytitle = "Weighted correlator"
p.NcolumnsLegend = 1
p.drawRatio = True
p.addBackground(h_truth_weighted, "Particle level (correlator-weighted)", ROOT.kAzure+7)
p.addSignal(h_bias_weighted, "Bias distribution", ROOT.kRed-2)
p.addSignal(h_truth_weighted_proj, "Particle level (bin-weighted)", ROOT.kGreen-2)
p.addData(h_unfolded_proj, "Unfolded")
p.draw()
