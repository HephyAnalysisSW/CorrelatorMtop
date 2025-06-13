#!/usr/bin/env python

import ROOT
import array
import os
import Analysis.Tools.syncer
import numpy as np
from math                           import sqrt
from MyRootTools.plotter.Plotter    import Plotter

ROOT.gROOT.SetBatch(ROOT.kTRUE)

def get_sample(sample_paths):
    combined_data = {}
    for sample_path in sample_paths:
        data_sample = np.load(sample_path)

        # Selection
        mask_sel = (data_sample['has_rec_info'] == 1) & \
               (data_sample['has_gen_info'] == 1) & \
               (data_sample['pass_triplet_top_gen'] == 1) & \
               (data_sample['pass_triplet_top_rec'] == 1)

        # Iterate over keys and collect selected data
        for key in data_sample.files:
            selected_data = data_sample[key][mask_sel]
            if key in combined_data:
                combined_data[key].append(selected_data)
            else:
                combined_data[key] = [selected_data]

    # Concatenate all arrays in the list for each key
    for key in combined_data:
        combined_data[key] = np.concatenate(combined_data[key])
    return combined_data

def fill_zeta_plot(hist, data, useRec = False):
    variables = ["zeta_rec", "zeta_weight_rec", "jetpt_rec"] if useRec else ["zeta_gen", "zeta_weight_gen", "jetpt_gen"]
    for zeta_gen, zeta_weight_gen, jetpt_gen in zip(data[variables[0]], data[variables[1]], data[variables[2]]):
        pt_weight_gen = jetpt_gen*jetpt_gen/(172.5*172.5)
        hist.Fill( zeta_gen*pt_weight_gen, zeta_weight_gen )

################################################################################
h_unfolded = ROOT.TH1F("unfolded", "", 7, 1.0, 7.0)
h_truth = ROOT.TH1F("truth", "", 7, 1.0, 7.0)
h_detector = ROOT.TH1F("detector", "", 7, 1.0, 7.0)

unfolded_path = "/groups/hephy/cms/dennis.schwarz/CorrelatorMtop/Unfolded/unfolded_v1.npz"
unfolded_data = np.load(unfolded_path)

fill_zeta_plot(h_unfolded, unfolded_data)

train_sample_list = ['/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/results/TTToSemiLeptonic_%i.npz'%(i) for i in range(0,50) ]
train_data = get_sample(train_sample_list)

fill_zeta_plot(h_truth, train_data)
fill_zeta_plot(h_detector, train_data, useRec=True)

h_unfolded.Scale(1/h_unfolded.Integral())
h_truth.Scale(1/h_truth.Integral())
h_detector.Scale(1/h_detector.Integral())


p = Plotter("Unfolded_CINN")
p.plot_dir = "/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/plots/Unfolding"
p.lumi = "60"
p.xtitle = "#it{p}_{T} weighted #zeta"
p.ytitle = "Weighted correlator"
p.NcolumnsLegend = 1
p.drawRatio = True
p.addBackground(h_truth, "Particle level", ROOT.kAzure+7)
p.addSignal(h_detector, "Detector level", ROOT.kRed-2)
p.addData(h_unfolded, "Unfolded")
p.draw()
