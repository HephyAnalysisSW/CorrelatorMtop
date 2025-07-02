# ML Unfolding

The ML unfolding comes in three basic steps:
1. producing a numpy file
2. train the model
3. use the model prediction for unfolding

## Producing numpy files

Two script take the flat ROOT TTree and write either a .npz or .h5, where the latter
is mostly used in the most recent unfolding class.
The scripts are `convert_to_hdf5.py` and `convert_to_npz.py`.
Both take a root file and create a corresponding npz or h5 file.
`combine_hdf5.py` can combine multiple .h5 file into one for better handling.
Corresponding .sh files show the respective commands to run the script.

## Unfolding

The most recent script is `unfold_v2.py` with the corresponding .sh file showing how to run it.
Two essential classes are used here: `H5Dataset` and `CINNUnfolding`, which are both defined in `Tools/python/`.

The `H5Dataset` class can handle .h5 files and provides the possibility to use a data loader.
This makes the handling in training a lot easier since batching can be used easily.

The `CINNUnfolding` class implements the cINN unfolding method.
