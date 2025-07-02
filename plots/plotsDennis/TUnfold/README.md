# Binned unfolding with TUnfold

## Compiling
For compiling the first time run:
```
make dict
make lib
make eeec
```

After changes to the user code it is sufficient to run
```
make eeec
```

## Run the code

The binned unfolding works in three steps:
1. Defining the binning
2. Filling histograms
3. Perform unfolding

### Binning

Run `EEEC_binning` to define the binning scheme.
It writes the binning scheme to a root/xml/txt file (e.g. `Binning.root`) that can later be read and interpreted by TUnfold.

### Filling histograms
Run `EEEC_histfiller` to fill histograms needed for the unfolding.
It produces the detector and particle level histograms as well as the response matrix that is later used in TUnfold.
For this, it reads the binning scheme from `Binning.root`.
The data is read from a flat ROOT TTree.
The Histograms are stored in a ROOT file: `EEEC_Histograms.root`

### Unfolding

Run `EEEC_unfold` to perform the unfolding.
There are three modes of regularization implemented that can be steered with a command line argument:
`fixedtau`: Use a fixed value for the regularization strength (which is hard-coded for now)
`lscan`: Use the L-curve scan implemented in TUnfold
`rhoscan`: Scan the global correlation coefficient as implemented in TUnfold

It reads the histograms from `EEEC_Histograms.root` and binning scheme from `Binning.root`.
It stores the unfolded distributions in a file `EEEC_Result.root`.

### Plotting
Two (preliminary) plotting scripts `EEEC_plot_input.py` and `EEEC_plot_unfolded.py` are
used for plotting the input distributions from `EEEC_Histograms.root` and unfolded results from `EEEC_Result.root`, respectively.
Both use the plotting class from https://github.com/denschwarz/MyRootTools/blob/main/plotter/python/Plotter.py
