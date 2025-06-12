print("Start of Script")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch import nn
from torch import optim
import matplotlib as mpl
import mplhep as hep
import sys
from datetime import datetime
import os
import plotutils
import transformations as trf
import psutil
import gc

from indices import (
    zeta_gen_index,
    zeta_rec_index,
    weight_gen_index,
    weight_rec_index,
    pt_gen_index,
    pt_rec_index,
    mass_gen_index,
    mass_jet_gen_index,
    gen_index,
    rec_index,
    zeta_sample_index,
    weight_sample_index,

    sample_index
)

from ml_functions import (calc_squared_weights,calculate_chisq)
#from MLUnfolding.Tools.user  import plot_directory
#plot_directory = "./plots"

from matplotlib.ticker import FuncFormatter

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel', action='store',      default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--train',  action='store', type=str, default="NA") # ./mldata/ML_Data_train.npy
argParser.add_argument('--val',    action='store', type=str, default="NA") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--plot_dir',    action='store', type=str, default="MLUnfolding_tmp") # ./mldata/ML_Data_validate.npy
argParser.add_argument('--load_model_file',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--save_model_path',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--load_model_path',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--info',    action='store', type=str, default="NA") #./mldata/ML_Data_validate.npy
argParser.add_argument('--weight_cut', action='store', type=float, default=0.0) # ./mldata/ML_Data_validate.npy
argParser.add_argument('--text_debug',    action='store', type=bool, default=True) #./mldata/ML_Data_validate.npy
argParser.add_argument('--shift', action='store', type=float, default=0.0) # ./mldata/ML_Data_validate.npy

args = argParser.parse_args()

gc.collect() #SH Test Garbage Collection
print(psutil.Process().memory_info().rss / (1024 * 1024))
print("Hi")

w_cut = args.weight_cut
text_debug= args.text_debug

plot_dir = os.path.join(args.plot_dir,"weight_cut_" + str(w_cut).replace('.', 'p'),"stack_with_sample_cut")# Zusammenpasten von plot_dir
print(plot_dir)
if not os.path.exists( plot_dir ): os.makedirs( plot_dir )
plot_dir_data = os.path.join(plot_dir,"data")
if not os.path.exists( plot_dir_data ): os.makedirs( plot_dir_data )

# the nflows functions what we will need in order to build our flow
from nflows.flows.base import Flow # a container that will wrap the parts that make up a normalizing flow
from nflows.distributions.normal import StandardNormal # Gaussian latent space distribution
from nflows.transforms.base import CompositeTransform # a wrapper to stack simpler transformations to form a more complex one
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform # the basic transformation, which we will stack several times
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform # the basic transformation, which we will stack several times
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.permutations import ReversePermutation # a layer that simply reverts the order of outputs

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(device)

text_debug= True#False
plot_debug = False#
table_debug = False#True#False


print("Started")
try :
    with open(args.train, "rb") as f:
        train_data_uncut = np.load(f)
        train_data = train_data_uncut[0:1000000]
        train_data = train_data[train_data[:, weight_gen_index] > w_cut] #SH: Apply cut for weight
        f.close()

except FileNotFoundError :
    print("File \""+ args.train+"\" (Train Data) not found.")
    exit(1)



train_data_lenght = np.shape(train_data)[0]
train_data_n_cols = np.shape(train_data)[1]

if text_debug == True :
    print("Lenght of training data: " + str(train_data_lenght))
    print("Cols of training data: "   + str(train_data_n_cols))

    #Print for
if table_debug == True :
    print("Imported Raw Training Data")
    print(train_data)

## transform data and save max, min, mean, std values for the backtransformation later
max_values = np.max(train_data, keepdims=True, axis=0)*1.1
min_values = np.min(train_data, keepdims=True, axis=0)/1.1

transformed_data, mask = trf.normalize_data(train_data, max_values, min_values)
transformed_data = trf.logit_data(transformed_data)
mean_values = np.mean(transformed_data, keepdims=True, axis=0)
std_values = np.std(transformed_data, keepdims=True, axis=0)
transformed_data = trf.standardize_data(transformed_data, mean_values, std_values)

print("\nSampling now:")

#SH: Redundant Exception Handling. Works though
try :
    with open(args.val, "rb") as f:
        val_data_uncut = np.load(f)
        val_data_plot = val_data_uncut[0:1000000]
        val_data = val_data_plot[val_data_plot[:, weight_gen_index] > w_cut] #SH: Apply cut for weight # val_data_plot#

        f.close()

except FileNotFoundError :
    print("File \""+ args.val+"\" not found.")
    exit(1)

val_transformed_data, mask = trf.normalize_data(val_data, max_values, min_values)
val_transformed_data = trf.logit_data(val_transformed_data)
val_transformed_data = trf.standardize_data(val_transformed_data, mean_values, std_values)
val_trans_cond = torch.tensor(val_transformed_data[:,rec_index], device=device).float()
val_data = val_data[mask]

print(val_trans_cond.shape)

# Get only .pt files
models = [file for file in os.listdir(args.load_model_path)
          if os.path.isfile(os.path.join(args.load_model_path, file)) and file.endswith('.pt')]

# Sort alphanumerically, handling natural numbers correctly
models.sort()

loss_function_in = []
loss_function_out=[]

#Calculate In Error
x_train = transformed_data[:,gen_index]
x_train = torch.tensor(x_train, device=device).float()
y_train = transformed_data[:,rec_index]
y_train = torch.tensor(y_train, device=device).float()

#Calculate Out Error
x_val = val_transformed_data[:,gen_index]
x_val = torch.tensor(x_val, device=device).float()
y_val = val_transformed_data[:,rec_index]
y_val = torch.tensor(y_val, device=device).float()

# PLOTTING STEERING
plt_w = 1 # SH: plot Weight
plt_wz = 2#SH: plot weighted Zeta
plt_z = 0#
plt_d = 0 #SH: Plot Data
plt_r = 1 #SH: Plot Ration Pad


#models = models[-10:] # only use last 10 models
#models = models[-1:] # only use last model

print(models)

for modelname in models:
    modelpath = args.load_model_path + "/"+ modelname

    try:
        flow =torch.load(modelpath, weights_only=False) # load flow from model
        flow.eval() # switch to evaluation mode
    except Exception as e:
        print(e)
        print("Not able to load given flow " + modelpath)
        exit(0)

    print("Sampling from flow " + modelpath)
    print(str(psutil.Process().memory_info().rss / (1024 * 1024)) + "MB")

    gc.collect() #SH Test Garbage Collection

    nll_in = -flow.log_prob(x_train, context=y_train) # Feed context
    loss_in = nll_in.mean()
    loss_function_in.append(loss_in.item())

    nll_out = -flow.log_prob(x_val, context=y_val)
    loss_out = nll_out.mean()
    loss_function_out.append(loss_out.item())

    with torch.no_grad():
      samples = flow.sample(1, context=val_trans_cond).view(val_trans_cond.shape[0], -1).cpu().numpy() # generate for a detector level condition one condition, this gives you the whole unfolded smaple!!
    ## inverse standardize
    retransformed_samples = trf.standardize_inverse(samples, mean_values[:,gen_index], std_values[:,gen_index])
    ## inverse logit
    retransformed_samples = trf.logit_inverse(retransformed_samples)
    ## inverse normalize
    retransformed_samples = trf.normalize_inverse(retransformed_samples, max_values[:,gen_index], min_values[:,gen_index])



    if text_debug :
        print("val data shape =",     val_trans_cond.shape)
        print("sampled data shape =", retransformed_samples.shape)

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")

    plotweight = np.shape(val_data)[0] / np.shape(train_data)[0]
    plotweights = np.full((np.shape(train_data)[0]),plotweight)

    modelname = modelname.replace(".pt", "")
    modelname = modelname.replace("m2f3e", "")
    modelname = modelname.zfill(2)

    #_________________________________________________________________________________________________________________
    #--SH: Start Plot in Mathplotlib
    fig, axs =  plt.subplots(2, 3, sharex = "col", tight_layout=True,figsize=(15, 6), gridspec_kw=
                                    dict(height_ratios=[6, 1],
                                          width_ratios=[1, 1, 1]))
    #SH: Add Ticks to every side
    for ax_row in axs:
        for ax in ax_row:
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

    #if args.info == "NA" :
    #    fig.suptitle("Epoch: "+modelname)
    #else :
    #    fig.suptitle(args.info + " | Epoch: "+modelname)


    #_________________________________________________________________________________________________________________
    #--SH: Plot Zeta`
    number_of_bins = 20
    upper_border = 7
    upper_border = upper_border *100
    step = upper_border // number_of_bins
    n_bins = [x / 100.0 for x in range(0,upper_border+1,step)]

    hist1,bin_edges = np.histogram(retransformed_samples[:,zeta_sample_index], bins= n_bins)
    hist2,_ = np.histogram(val_data[:,zeta_rec_index]    ,bins= n_bins)
    hist3,_ = np.histogram(val_data[:,zeta_gen_index] ,bins= n_bins)
    hist4 = np.divide(hist1, hist3, where=hist3!=0)

    hep.histplot(hist1,       n_bins, ax=axs[plt_d,plt_z],color = "red",alpha = 1 ,      label = "ML Val Particle Lvl")#, histtype="fill")
    hep.histplot(hist2,       n_bins, ax=axs[plt_d,plt_z],color = "black",   label = "Val Detector Lvl")
    hep.histplot(hist3,       n_bins, ax=axs[plt_d,plt_z],color = "#999999", label = "Particle Lvl"    )
    hep.histplot(hist4, n_bins, ax=axs[plt_r,plt_z],color = "red", alpha = 0.5)

    #SH get plot weights
    sample_plot_weight = retransformed_samples[:, weight_sample_index]
    val_gen_plot_weight = val_data[:, weight_gen_index]
    val_rec_plot_weight = val_data[:, weight_rec_index]
    train_gen_plot_weight = train_data[:, weight_gen_index]
    val_plt_gen_plot_weight = val_data_plot[:,weight_gen_index] # Ohne weight cut

    sample_plot_weight = sample_plot_weight / np.sum(sample_plot_weight)
    val_gen_plot_weight = val_gen_plot_weight / np.sum(val_gen_plot_weight)
    val_rec_plot_weight = val_rec_plot_weight / np.sum(val_rec_plot_weight)
    train_gen_plot_weight = train_gen_plot_weight / np.sum(train_gen_plot_weight)
    val_plt_gen_plot_weight = val_plt_gen_plot_weight / np.sum(val_plt_gen_plot_weight)

    #_________________________________________________________________________________________________________________
    #--SH: Plot Weight
    upper_border = 1000 #300
    step = upper_border // number_of_bins
    n_bins = [x / 10000000.0 for x in range(0,upper_border+1,step)]


    hist1,_ = np.histogram(retransformed_samples[:,weight_sample_index], bins= n_bins)
    hist2,_ = np.histogram(val_data[:,weight_rec_index]    ,bins= n_bins)
    hist3,_ = np.histogram(val_data[:,weight_gen_index] , bins= n_bins)
    hist4 = np.divide(hist1, hist3, where=hist3!=0)

    hep.histplot(hist1,       n_bins, ax=axs[plt_d,plt_w],color = "red",alpha = 1,      label = "ML Val Particle Lvl")#, histtype="fill")
    hep.histplot(hist2,       n_bins, ax=axs[plt_d,plt_w],color = "black",   label = "Val Detector Lvl")
    hep.histplot(hist3,       n_bins, ax=axs[plt_d,plt_w],color = "#999999", label = "Particle Lvl"    )
    hep.histplot(hist4, n_bins, ax=axs[plt_r,plt_w],color = "red", alpha = 0.5)

    #_________________________________________________________________________________________________________________
    #--SH: Plot Weighted Zeta'
    upper_border = 7
    upper_border = upper_border *100
    step = upper_border // number_of_bins
    n_bins = [x / 100.0 for x in range(0,upper_border+1,step)]

    hist1, bin_edges = np.histogram(retransformed_samples[:,zeta_sample_index] , weights= sample_plot_weight , bins= n_bins)
    hist2,_ = np.histogram(val_data[:,zeta_rec_index]  , weights= val_rec_plot_weight   , bins= n_bins)
    hist3,_ = np.histogram(val_data[:,zeta_gen_index]  , weights= val_gen_plot_weight   , bins= n_bins)
    hist4 = np.divide(hist1, hist3, where=hist3!=0)
    hist5,_ = np.histogram(train_data[:,zeta_gen_index] , weights= train_gen_plot_weight  , bins= n_bins)
    hist6,_ = np.histogram(val_data_plot[:,zeta_gen_index]  , weights=  val_plt_gen_plot_weight  , bins= n_bins)

    hist1_error, _ = np.histogram(retransformed_samples[:,zeta_sample_index] , weights= sample_plot_weight**2 , bins= n_bins)
    hist1_error = np.sqrt(hist1_error)
    hist2_error, _ = np.histogram(val_data[:,zeta_rec_index]  , weights= val_rec_plot_weight**2   , bins= n_bins)
    hist2_error = np.sqrt(hist2_error)
    hist3_error, _ = np.histogram(val_data[:,zeta_gen_index]  , weights= val_gen_plot_weight**2   , bins= n_bins)
    hist3_error = np.sqrt(hist3_error)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    hep.histplot(hist1,       n_bins, ax=axs[plt_d,plt_wz],color = "red",alpha = 1)#,histtype="fill",label = "ML Val Particle Lvl")
    hep.histplot(hist2,       n_bins, ax=axs[plt_d,plt_wz],color = "black",   label = "Val Detector Lvl")
    hep.histplot(hist3,       n_bins, ax=axs[plt_d,plt_wz],color = "#999999", label = "Particle Lvl"    ) # grau
    #hep.histplot(hist5,       n_bins, ax=axs[plt_d,plt_wz],color = "#0352fc", label = "Train Particle Lvl"    ) # Blue comparison Histogramm
    #hep.histplot(hist6,       n_bins, ax=axs[plt_d,plt_wz],color = "green",alpha = 0.7, label = "Val Particle Lvl wo Filter") # Green comparison Histogramm
    hep.histplot(hist4, n_bins, ax=axs[plt_r,plt_wz],color = "red", alpha = 0.5)

    axs[plt_d,plt_wz].errorbar(bin_centers, hist1, yerr=hist1_error, fmt='none', ecolor='red', alpha=0.5, capsize=5, capthick=1)     # Add error bars

    weight_squared = calc_squared_weights(retransformed_samples[:,zeta_sample_index] , weights= sample_plot_weight , bins= n_bins)
    chi2 = calculate_chisq(hist1,hist3,weight_squared)

    print("Chi^2: " + str(round(chi2,3)))

    #_________________________________________________________________________________________________________________
    #--SH: Plot Style and Axis
    axs[plt_d,plt_z].set_yscale("log")
    axs[plt_d,plt_w].set_yscale("log")

    #axs[plt_d,plt_z].legend(frameon = False, fontsize="18")
    axs[plt_d,plt_w].legend(frameon = False, fontsize="18")
    #axs[plt_d,plt_wz].legend(frameon = False, fontsize="14", loc=8)

    axs[plt_r,plt_z].set_ylabel(" $\\frac{\\mathrm{ML Particle Lvl}}{\\mathrm{Val Particle Lvl}}$")
    axs[plt_r,plt_w].set_ylabel(" $\\frac{\\mathrm{ML Particle Lvl}}{\\mathrm{Val Particle Lvl}}$")
    axs[plt_r,plt_wz].set_ylabel(" $\\frac{\\mathrm{ML Particle Lvl}}{\\mathrm{Val Particle Lvl}}$")

    axs[plt_r,plt_z].grid(axis = "y")
    axs[plt_r,plt_w].grid(axis = "y")
    axs[plt_r,plt_wz].grid(axis = "y")

    axs[plt_r,plt_z].set_ylim([0.7, +1.3])
    axs[plt_r,plt_w].set_ylim([0.7, +1.3])
    axs[plt_r,plt_wz].set_ylim([0.7, +1.3])

    formatter = FuncFormatter(lambda x, _: f'{x:.0e}')
    axs[plt_r,plt_w].xaxis.set_major_formatter(formatter)

    axs[plt_d,plt_z].set_ylim([1, 1e5])
    axs[plt_d,plt_w].set_ylim([1e3, 1e5])
    axs[plt_d,plt_z].set_xlim([0, 7])
    axs[plt_d,plt_w].set_xlim([0, 0.0001])
    #axs[0,plt_w].set_xlim([0, 0.00003])
    axs[plt_d,plt_wz].set_xlim([0, 7])

    xlabels = [
        r"$\zeta$ * pt$^2$ / 172.5$^2$",
        r"weight ($ \prod \frac{p_i}{p_t}$)",
        r"$\zeta$ * pt$^2$ / 172.5$^2$"
    ]
    ylabels = ["Events","Events","Events (weighted)"]

    for col in range(3):
        axs[plt_r, col].text(
            1.02, -0.4,
            xlabels[col],
            transform=axs[1, col].transAxes,
            fontsize=16,
            va="top", ha="right"
        )
        axs[plt_d, col].text(
            -0.1, 1.0,
            ylabels[col],
            transform=axs[0, col].transAxes,
            fontsize=16,
            va="top", ha="right",
            rotation=90
        )

    plt.savefig(plot_dir+"/generated_data_"+modelname+".png")
    plt.close("all")

    with open(plot_dir_data+"/"+modelname+".npy", 'wb') as f0:
        np.save(f0, retransformed_samples )

with open(plot_dir_data+"/train.npy", 'wb') as f0:
    np.save(f0, train_data_uncut )
with open(plot_dir_data+"/val.npy", 'wb') as f1:
    np.save(f1, val_data_uncut )

it=[*range(len(loss_function_in))]

plt.plot(it,loss_function_in, label="Train-Loss", color="#696969")
plt.plot(it,loss_function_out, label="Validation-Loss", color="red", alpha=0.5)
plt.xlabel("Epochs")
plt.ylabel("-log loss")
plt.legend()
plt.savefig(plot_dir_data+"/loss"+current_time+".png")
plt.close("all")
