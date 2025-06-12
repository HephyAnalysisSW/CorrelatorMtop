import ROOT
import argparse
import numpy as np
from RootTools.core.Sample import Sample
from tqdm import tqdm

def main(filename):
    # Define which variables to read from root tree (and later save in .npz)
    read_variables = [
        "ievent/I",
        "zeta_weight_rec/F",
        "zeta_weight_gen/F ",
        "zeta_rec/F",
        "zeta_gen/F",
        "jetpt_rec/F",
        "jetpt_gen/F",
        "has_gen_info/I",
        "has_rec_info/I",
        "pass_triplet_top_rec/I",
        "pass_triplet_top_gen/I",
        "BW_reweight_171p5/F",
        "BW_reweight_173p5/F",
        "mtop/F",
        "event_weight_gen/F",
        "event_weight_rec/F",
    ]

    # Create arrays for .npz
    # these are python lists for now
    arrays = {}
    for var in read_variables:
        varname = var.split("/")[0]
        arrays[varname] = []

    # define a sample from a single root file
    sample = Sample.fromFiles(name="sample", files = [filename], treeName="Events")
    r = sample.treeReader( variables = read_variables )
    r.start()
    nEvents = r.nEvents

    # Loop over tree and append in arrays
    print("Processing a total of", nEvents, "entries")
    progressbar = tqdm(total=nEvents, desc="Processing")
    while r.run():
        event = r.event
        for var in read_variables:
            varname = var.split("/")[0]
            value = getattr(event, varname)
            arrays[varname].append(value)
        progressbar.update(1)
    progressbar.close()

    # Convert all values to NumPy arrays and save
    outname = filename.replace(".root", ".npz")
    np.savez(outname, **{k: np.array(v) for k, v in arrays.items()})
    print("Numpy file saved:", outname)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--logLevel',       action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
    argParser.add_argument('--file',           action='store', default=None)
    args = argParser.parse_args()

    main(args.file)
