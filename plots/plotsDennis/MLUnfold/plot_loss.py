import os
import numpy as np
import argparse
import array
import ROOT
import Analysis.Tools.syncer
from CorrelatorMtop.Tools.user import plot_directory

def getGraph(values):
    x_values = []
    y_values = []
    for i, y in enumerate(values):
        x_values.append(i+1)
        y_values.append(y)

    graph = ROOT.TGraph(len(x_values), array.array('d', x_values) , array.array('d', y_values) )
    return graph

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--file', action='store')
args = argParser.parse_args()

data = np.load(args.file)
g_loss_train = getGraph(data['loss_train'])
g_loss_val = getGraph(data['loss_val'])

plotpath = os.path.join(plot_directory, "Unfolding", "loss.pdf")
c = ROOT.TCanvas("", "", 600, 600)
g_loss_train.SetLineColor(ROOT.kAzure+7)
g_loss_val.SetLineColor(ROOT.kRed-2)
g_loss_train.Draw("AL")
g_loss_val.Draw("L SAME")
c.Print(plotpath)
