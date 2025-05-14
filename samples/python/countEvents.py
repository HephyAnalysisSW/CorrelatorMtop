import ROOT
import sys
from CorrelatorMtop.samples.UL2018_unprocessed import allSamples

def print_progress_bar(processed, total, bar_length=40):
    percent = float(processed) / total
    filled_length = int(bar_length * percent)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r[{0}] {1:6.2f}% ({2}/{3})'.format(bar, percent * 100, processed, total))
    sys.stdout.flush()
    if processed == total:
        print()  # new line when complete
        
def sumWeights(file_path, treename):
    file = ROOT.TFile.Open(file_path)
    tree = file.Get(treename)
    generator_weight = ROOT.std.vector('float')()
    sum_weights = 0.0
    for entry in tree:
        sum_weights += entry.Generator_weight
    return sum_weights

for sample in allSamples:
    print("Counting events in %s"%(sample.name))
    weight_sum = 0
    for i,fname in enumerate(sample.files):
        print_progress_bar(i+1, len(sample.files))
        weight_sum += sumWeights(fname, "Events_All")
    print("%s: %.4f events"%(sample.name, weight_sum))
