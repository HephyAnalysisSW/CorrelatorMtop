#!/usr/bin/env python

import ROOT
import array
import os
import Analysis.Tools.syncer
import numpy as np
from math                           import sqrt
from MyRootTools.plotter.Plotter    import Plotter

ROOT.gROOT.SetBatch(ROOT.kTRUE)

def fill_zeta_plot(hist, data):
    for zeta_gen, zeta_weight_gen, jetpt_gen in zip(data["zeta_gen"], data["zeta_weight_gen"], data["jetpt_gen"]):
        pt_weight_gen = jetpt_gen*jetpt_gen/(172.5*172.5)
        hist.Fill( zeta_gen*pt_weight_gen, zeta_weight_gen )

unfolded_path = "/groups/hephy/cms/dennis.schwarz/CorrelatorMtop/Unfolded/unfolded_v1.npz"
unfolded_data = np.load(unfolded_path)

h_unfolded = ROOT.TH1F("unfolded", "", 7, 1.0, 7.0)
fill_zeta_plot(h_unfolded, unfolded_data)

p = Plotter("Unfolded_CINN")
p.plot_dir = "/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/plots/Unfolding"
p.lumi = "60"
p.xtitle = "#it{p}_{T} weighted #zeta"
p.ytitle = "Weighted correlator"
p.NcolumnsLegend = 1
p.drawRatio = True
# p.addBackground(h_truth, "Particle level", ROOT.kAzure+7)
# p.addSignal(h_bias, "Bias distribution", ROOT.kRed-2)
p.addData(h_unfolded, "Unfolded")
p.draw()
