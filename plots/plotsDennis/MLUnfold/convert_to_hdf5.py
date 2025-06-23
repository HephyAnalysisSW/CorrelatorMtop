import h5py
import numpy as np
import ROOT
import argparse
from RootTools.core.Sample import Sample
from tqdm import tqdm

def main(filename, small):
    # Define which variables to read from root tree (and later save in .h5)
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

    # Create arrays for .h5
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
    maxEvents = None if not small else 1000
    i_event = 0
    while r.run():
        i_event += 1
        if maxEvents is not None and i_event > maxEvents:
            break
        event = r.event
        for var in read_variables:
            varname = var.split("/")[0]
            value = getattr(event, varname)
            arrays[varname].append(value)
        progressbar.update(1)
    progressbar.close()

    # Convert all values to H5 and save
    outname = filename.split("/")[-1].replace(".root", ".h5")
    outname = "/scratch-cbe/users/dennis.schwarz/MTopCorrelations_h5/"+outname
    if small:
        outname = outname.replace(".h5", "_small.h5")
    print(outname)
    with h5py.File(outname, "w") as h5file:
        h5file.swmr_mode = True
        for varname in arrays.keys():
            print(varname)
            h5file.create_dataset(varname, data=arrays[varname], compression="gzip")

    print("H5 file saved:", outname)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--logLevel',       action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
    argParser.add_argument('--file',           action='store', default=None)
    argParser.add_argument('--small',           action='store_true', default=False)
    args = argParser.parse_args()

    main(args.file, args.small)
