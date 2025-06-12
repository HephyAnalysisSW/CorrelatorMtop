import numpy as np

## defining variable transformation und retransformation
def normalize_data(in_data, max_val, min_val):
    new_data = (in_data-min_val)/(max_val-min_val)
    mask = np.prod(((new_data < 1) & (new_data > 0 )), axis=1, dtype=bool)
    new_data = new_data[mask]
    return new_data, mask

def logit_data(in_data):
    new_data = np.log(in_data/(1-in_data))
    return new_data

def standardize_data(in_data, mean_val, std_val):
    new_data = (in_data - mean_val)/std_val
    return new_data

## defining their inverse transformations
def normalize_inverse(in_data, max_val, min_val):
    new_data = in_data*(max_val-min_val) + min_val
    return new_data

def logit_inverse(in_data):
    new_data = (1+np.exp(-in_data))**(-1)
    return new_data

def standardize_inverse(in_data, mean_val, std_val):
    new_data = std_val*in_data + mean_val
    return new_data
