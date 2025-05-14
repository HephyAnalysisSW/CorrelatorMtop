#!/usr/bin/env python

import ROOT
import array
import os
import Analysis.Tools.syncer

from CorrelatorMtop.Tools.helpers   import getObjFromFile
from MyRootTools.plotter.Plotter    import Plotter

ROOT.gROOT.SetBatch(ROOT.kTRUE)

colors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 28, 30, 32, 38, 41, 42, 46, 49, 50, 52, 55]

def getContents(filename):
    contents = []
    # Loop over all keys in the file
    root_file = ROOT.TFile.Open(filename)
    keys = root_file.GetListOfKeys()
    for key in keys:
        obj = key.ReadObj()
        # Check if the object inherits from TH1 (includes TH1F, TH2F, etc.)
        if obj.InheritsFrom("TH1"):
            hist_type = obj.ClassName()
            hist_name = obj.GetName()
            contents.append( (hist_name, hist_type) )
    root_file.Close()
    return contents

def drawMatrix(hist, outname):
    ROOT.gStyle.SetLegendBorderSize(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPalette(ROOT.kSunset)
    c = ROOT.TCanvas("c", "c", 600, 600)
    ROOT.gPad.SetRightMargin(0.19)
    ROOT.gPad.SetLeftMargin(0.19)
    ROOT.gPad.SetBottomMargin(0.12)
    hist.Draw("COLZ")
    c.Print(outname)

plotdir = "/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/plots/Unfolding"

filename = "EEEC_Histograms.root"
contents = getContents(filename)

hist_collections = {}

for (hname, htype) in contents:
    hist = getObjFromFile(filename, hname)
    if "TH2" in htype:
        print("Draw %s as 2D hist"%(hname))
        drawMatrix(hist, plotdir+"/"+hname+".pdf")
    else:
        print("Draw %s as 1D hist"%(hname))
        p = Plotter(hname)
        p.plot_dir = plotdir
        p.lumi = "60"
        p.xtitle = ""
        p.ytitle = ""
        p.NcolumnsLegend = 1
        p.addSignal(hist, hname, ROOT.kAzure+7)
        p.draw()
        # Also sort into collections
        collection = hname.split("__")[0]
        if collection not in hist_collections.keys():
            hist_collections[collection] = [(hname, hist)]
        else:
            hist_collections[collection].append( (hname, hist) )

for collection in hist_collections.keys():
    p = Plotter(collection)
    p.plot_dir = plotdir
    p.lumi = "60"
    p.xtitle = ""
    p.ytitle = ""
    p.drawRatio = True
    p.NcolumnsLegend = 1
    for i, (hname, hist) in enumerate(hist_collections[collection]):
        if "ttbar" in hname:
            p.addBackground(hist, hname.replace(collection+"__", ""), 15)
        else:
            p.addSignal(hist, hname.replace(collection+"__", ""), colors[i])
    p.draw()
