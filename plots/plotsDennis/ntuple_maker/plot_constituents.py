#!/usr/bin/env python

import ROOT
import array
import os
import Analysis.Tools.syncer
import numpy as np
from math                           import sqrt
from MyRootTools.plotter.Plotter    import Plotter
from CorrelatorMtop.Tools.helpers   import getObjFromFile

ROOT.gROOT.SetBatch(ROOT.kTRUE)


file_name = "/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/results/TTToSemiLeptonic_HISTS.root"

for genrec in ["gen", "rec"]:
    h_selection = getObjFromFile(file_name, "N_const_"+genrec)
    h_all = getObjFromFile(file_name, "N_const_nocut_"+genrec)
    h_onlypt = getObjFromFile(file_name, "N_const_onlypt_"+genrec)

    p = Plotter("Nconstituents_"+genrec)
    p.plot_dir = "/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/plots/Ntuple"
    p.lumi = "60"
    p.xtitle = "Number of charged constituents"
    p.ytitle = "Events"
    p.NcolumnsLegend = 1
    p.legtextsize = 0.05
    p.drawRatio = False
    p.addSignal(h_all, "No selection", ROOT.kRed-2)
    p.addSignal(h_onlypt, "#it{p}_{T} > 5 GeV", ROOT.kAzure+7)
    # p.addSignal(h_selection, "#it{p}_{T} > 5 GeV, max(N_{const}) = 25", 15)
    p.draw()


for histname in ["particle_matching_effi", "triplet_matching_effi_gen", "triplet_matching_effi_rec"]:
    hist = getObjFromFile(file_name, histname)
    p = Plotter(histname)
    p.plot_dir = "/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/plots/Ntuple"
    p.lumi = "60"
    p.xtitle = "Efficiency"
    p.ytitle = "Events"
    p.NcolumnsLegend = 1
    p.legtextsize = 0.05
    p.drawRatio = False
    p.addSignal(hist, histname.replace("_", " "), ROOT.kAzure+7)
    p.draw()
