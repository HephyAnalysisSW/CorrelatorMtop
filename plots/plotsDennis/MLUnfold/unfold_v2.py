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
from CorrelatorMtop.Tools.H5Dataset       import H5Dataset
from CorrelatorMtop.Tools.CINNUnfolding   import CINNUnfolding


################################################################################
################################################################################
################################################################################

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--small', action='store_true')
argParser.add_argument('--train', action='store_true')
argParser.add_argument('--predict', action='store_true')
argParser.add_argument('--reduce', action='store', default=None, type=int)

args = argParser.parse_args()

################################################################################
# Logger
import CorrelatorMtop.Tools.logger as logger
import RootTools.core.logger as logger_rt
logger    = logger.get_logger(   args.logLevel, logFile = None)
logger_rt = logger_rt.get_logger(args.logLevel, logFile = None)

if args.train:
    logger.info("Starting training mode")
if args.predict:
    logger.info("Starting prediction mode")
if args.small:
    logger.info("Selected 'small', only running a fraction of the data")

################################################################################
# Define samples

logger.info("Defining data samples")

# Features for unfolding
rec_features = ["zeta_rec", "zeta_weight_rec", "jetpt_rec"]
gen_features = ["zeta_gen", "zeta_weight_gen", "jetpt_gen"]

# Define selection string
selection_string = "(has_rec_info == 1) & (has_gen_info == 1) & (pass_triplet_top_gen == 1) & (pass_triplet_top_rec == 1)"

# Path to h5 files
train_file_path = "/scratch-cbe/users/dennis.schwarz/MTopCorrelations_h5/TTToSemiLeptonic_train.h5"
val_file_path   = "/scratch-cbe/users/dennis.schwarz/MTopCorrelations_h5/TTToSemiLeptonic_val.h5"

# create data sets
fraction = 1.0
if args.small:
    fraction = 0.0001
if args.reduce is not None:
    fraction = 1.0/args.reduce
train_dataset = H5Dataset(train_file_path, gen_features, rec_features, selection=selection_string, fraction=fraction)
val_dataset = H5Dataset(train_file_path, gen_features, rec_features, selection=selection_string, fraction=fraction)


################################################################################
# Unfolding
logger.info("Set up unfolding class")
model_path = "/groups/hephy/cms/dennis.schwarz/CorrelatorMtop/Unfolding_model_v2/"
if args.small:
    model_path = "/groups/hephy/cms/dennis.schwarz/CorrelatorMtop/Unfolding_model_v2_small/"

model = CINNUnfolding(
    train_data=train_dataset,
    val_data=val_dataset,
    rec_features=rec_features,
    gen_features=gen_features,
    save_path=model_path,
    normalize = True
)

### TODO
# one could make the min/max parameters for normalization part of the model and save those after training
# then those could be read for predict

model.logger = logger
model.n_epochs = 200
model.learning_rate = 1e-5
# model.batch_size = 5000




if args.train:
    logger.info("Start training")
    model.train()

# Later:
# samples = model.predict("val_data.h5", "./cinn_checkpoints/model_epoch25.pt", n_samples=1)
