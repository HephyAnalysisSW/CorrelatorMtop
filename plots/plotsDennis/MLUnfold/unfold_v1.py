import numpy as np
import torch
from torch import nn
from torch import optim
import argparse
import sys
import gc
from datetime import datetime
import os
import psutil
from CorrelatorMtop.Tools.cinn_unfolding import cinn_unfolding

################################################################################
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

################################################################################
################################################################################
################################################################################

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--train', action='store_true')
argParser.add_argument('--predict', action='store_true')
args = argParser.parse_args()

# Logger
import CorrelatorMtop.Tools.logger as logger
import RootTools.core.logger as logger_rt
logger    = logger.get_logger(   args.logLevel, logFile = None)
logger_rt = logger_rt.get_logger(args.logLevel, logFile = None)

# Load data for training
logger.info("Loading samples...")

# train_sample_list = [f'/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/results/TTToSemiLeptonic_{i}.npz' for i in range(0,50) ]
# validation_sample_list = [f'/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/results/TTToSemiLeptonic_{i}.npz' for i in range(50,90) ]
# pseudodata_sample_list = [f'/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/results/TTToSemiLeptonic_{i}.npz' for i in range(90,100) ]
train_sample_list = [f'/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/results/TTToSemiLeptonic_{i}.npz' for i in range(0,50) ]
validation_sample_list = [f'/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/results/TTToSemiLeptonic_{i}.npz' for i in range(50,60) ]
pseudodata_sample_list = [f'/groups/hephy/cms/dennis.schwarz/www/CorrelatorMtop/results/TTToSemiLeptonic_{i}.npz' for i in range(90,91) ]


train_data_transformed = get_sample(train_sample_list)
validation_data_transformed = get_sample(validation_sample_list)
logger.info("Samples loaded.")

# Train
logger.info("Set up unfolding model")
model_save_path = "/groups/hephy/cms/dennis.schwarz/CorrelatorMtop/Unfolding_model_v1"
unfolding_model = cinn_unfolding(
    train_data = train_data_transformed,
    validation_data = validation_data_transformed,
    gen_variables = ["zeta_gen", "zeta_weight_gen", "jetpt_gen"],
    rec_variables = ["zeta_rec", "zeta_weight_rec", "jetpt_rec"],
    save_path = model_save_path
    )
unfolding_model.n_epochs = 100
unfolding_model.set_logger(logger)
unfolding_model.prepare()
if args.train:
    logger.info("Start training...")
    unfolding_model.train()
if args.predict:
    logger.info("Start predicting...")
    pseudodata_transformed = get_sample(pseudodata_sample_list)
    unfolding_model.modelpath = "/groups/hephy/cms/dennis.schwarz/CorrelatorMtop/Unfolding_model_v1/m2f3e100of100.pt"
    unfolded_sample = unfolding_model.predict(pseudodata_transformed)

    zeta_gen          = unfolded_sample[:, 0]
    zeta_weight_gen   = unfolded_sample[:, 1]
    jetpt_gen         = unfolded_sample[:, 2]
    unfolded_ntuple_path = "/groups/hephy/cms/dennis.schwarz/CorrelatorMtop/Unfolded/unfolded_v1.npz"
    np.savez(unfolded_ntuple_path, zeta_gen=zeta_gen, zeta_weight_gen=zeta_weight_gen, jetpt_gen=jetpt_gen)
    logger.info(f"Saved unfolded ntuple: {unfolded_ntuple_path}")
logger.info("Done.")
