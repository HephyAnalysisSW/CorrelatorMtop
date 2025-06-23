import numpy as np
import torch
from torch import nn
from torch import optim
import argparse
import sys
import gc
import os
import psutil
import CorrelatorMtop.Tools.transformations as trf

# the nflows functions what we will need in order to build our flow
from nflows.flows.base import Flow # a container that will wrap the parts that make up a normalizing flow
from nflows.distributions.normal import StandardNormal # Gaussian latent space distribution
from nflows.transforms.base import CompositeTransform # a wrapper to stack simpler transformations to form a more complex one
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform # the basic transformation, which we will stack several times
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform # the basic transformation, which we will stack several times
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.permutations import ReversePermutation # a layer that simply reverts the order of outputs



class cinn_unfolding:
    def __init__(self, train_data, validation_data, gen_variables, rec_variables, save_path):
        self.train_data = train_data
        self.validation_data = validation_data
        self.gen_features = gen_variables
        self.rec_features = rec_variables
        self.save_path = save_path
        self.gen_index = None
        self.rec_index = None
        self.train_stacked = None
        self.validation_stacked = None
        self.max_values = None
        self.min_values = None
        self.modelpath = None
        self.isReady = False

        # Nflow parameters
        self.n_features = len(gen_variables)
        self.n_features_con = len(rec_variables)
        self.n_layers = 6 # number of transformations in the flow

        # training parameters
        self.learning_rate = 0.00001
        self.n_epochs = 25
        self.batch_size =  128
        self.model_id = 2

        # store loss
        self.loss_function_train = []
        self.loss_function_validation = []

        # gpu or cpu?
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0") if cuda else torch.device("cpu")

        # logger
        self.logger = None


    ############################################################################
    # setup the flow
    def __get_flow(self):
        base_dist = StandardNormal(shape=[self.n_features])
        transforms = []
        for i in range(0, self.n_layers):
            transforms.append(MaskedAffineAutoregressiveTransform(features=self.n_features, hidden_features=32, context_features=self.n_features_con))
            transforms.append(ReversePermutation(features=self.n_features))
        transform = CompositeTransform(transforms)
        flow = Flow(transform, base_dist).to(self.device)
        return flow

    ############################################################################
    # convert sample into numpy array
    def __stack_sample(self, sample):
        index_dict = {}
        list_of_arrays = []
        all_features = self.gen_features+self.rec_features
        for i,feature in enumerate(all_features):
            list_of_arrays.append( sample[feature] )
            index_dict[feature] = i

        # make np array
        sample_stacked = np.stack(list_of_arrays).T

        # also store position of gen/rec features
        gen_index = [index_dict[feature] for feature in self.gen_features]
        rec_index = [index_dict[feature] for feature in self.rec_features]
        return sample_stacked, gen_index, rec_index

    ############################################################################
    # set logger
    def set_logger(self, logger):
        self.logger = logger

    ############################################################################
    # returns loss values
    def get_loss(self):
        return self.loss_function_train, self.loss_function_validation

    ############################################################################
    # store min and max values of training sample
    # these are used for each sample to transform the data, thus the min/max
    # values need to be stored
    def prepare(self):
        self.train_stacked, self.gen_index, self.rec_index = self.__stack_sample(self.train_data)
        self.validation_stacked, _, _ = self.__stack_sample(self.validation_data)

        self.max_values = np.max(self.train_stacked, keepdims=True, axis=0)*1.1
        self.min_values = np.min(self.train_stacked, keepdims=True, axis=0)/1.1
        self.isReady = True

    ############################################################################
    # helper for transforming data using min and max values
    def transform_data(self, sample):
        sample_transformed, mask = trf.normalize_data(sample, self.max_values, self.min_values)
        sample_transformed = trf.logit_data(sample_transformed)
        mean_values = np.mean(sample_transformed, keepdims=True, axis=0)
        std_values = np.std(sample_transformed, keepdims=True, axis=0)
        sample_transformed = trf.standardize_data(sample_transformed, mean_values, std_values)
        return sample_transformed, mask, mean_values, std_values

    ############################################################################
    # training
    def train(self):
        if not self.isReady:
            raise Exception("Need to run cinn_unfolding.prepare() first!")

        train_transformed, _, _, _ = self.transform_data(self.train_stacked)
        validation_transformed, _, _, _ = self.transform_data(self.validation_stacked)

        flow = self.__get_flow()
        optimizer = optim.Adam(flow.parameters(), lr=self.learning_rate)

        max_batches = int(train_transformed.shape[0] / self.batch_size)
        save_model_path = os.path.join(self.save_path)
        if not os.path.exists( save_model_path ): os.makedirs( save_model_path )
        save_model_file = save_model_path+"/m"+str(self.model_id)+"f"+str(self.n_features)+"e"+"00of"+str(self.n_epochs)+".pt"
        torch.save(flow, save_model_file)

        for i in range(self.n_epochs):
            permut = np.random.permutation(train_transformed.shape[0])
            train_shuffle = train_transformed[permut]

            gc.collect() #SH Test Garbage Collection
            if self.logger:
                self.logger.info("Epoch "+str(i+1)+"/"+str(self.n_epochs))
            else:
                print("Epoch "+str(i+1)+"/"+str(self.n_epochs))

            for i_batch in range(max_batches):
                x = train_shuffle[i_batch*self.batch_size:(i_batch+1)*self.batch_size,self.gen_index]
                x = torch.tensor(x, device=self.device).float()

                y = train_shuffle[i_batch*self.batch_size:(i_batch+1)*self.batch_size,self.rec_index]
                y = torch.tensor(y, device=self.device).float()#.view(-1, 1)

                optimizer.zero_grad()
                nll = -flow.log_prob(x, context=y) # Feed context
                loss = nll.mean()

                loss.backward()
                optimizer.step()
                del x
                del y
                del nll
                del loss
                gc.collect() #SH Test Garbage Collection

            #Calculate In Error
            x_train = train_shuffle[:,self.gen_index]
            x_train = torch.tensor(x_train, device=self.device).float()
            y_train = train_shuffle[:,self.rec_index]
            y_train = torch.tensor(y_train, device=self.device).float()

            nll_in = -flow.log_prob(x_train, context=y_train)
            loss_in = nll_in.mean()
            self.loss_function_train.append(loss_in.item())

            #Calculate Out Error
            x_val = validation_transformed[:,self.gen_index]
            x_val = torch.tensor(x_val, device=self.device).float()
            y_val = validation_transformed[:,self.rec_index]
            y_val = torch.tensor(y_val, device=self.device).float()

            nll_out = -flow.log_prob(x_val, context=y_val)
            loss_out = nll_out.mean()
            self.loss_function_validation.append(loss_out.item())

            save_model_file = save_model_path+"/m"+str(self.model_id)+"f"+str(self.n_features)+"e"+str(i+1).zfill(2)+"of"+str(self.n_epochs)+".pt" # 3of50 = after the 3rd training
            torch.save(flow, save_model_file)
        print(f"Saved model: {save_model_file}")
        loss_file_name = save_model_file.replace(".pt", "__loss.npz")
        np.savez(loss_file_name,
            loss_train=self.loss_function_train,
            loss_validation=self.loss_function_validation,
            )
        print(f"Saved loss function: {loss_file_name}")
        gc.collect()

    ############################################################################
    # predict function for performing the unfolding
    def predict(self, sample):
        if not self.isReady:
            raise Exception("Need to run cinn_unfolding.prepare() first!")

        sample_stacked, _, _ = self.__stack_sample(sample)
        validation_transformed, _, mean_values, std_values = self.transform_data(sample_stacked)
        val_trans_cond = torch.tensor(validation_transformed[:,self.rec_index], device=self.device).float()

        try:
            flow_loaded = torch.load(self.modelpath) # load flow from model
            # flow_loaded = torch.load(self.modelpath, weights_only=False) # load flow from model
            flow_loaded.eval() # switch to evaluation mode
        except Exception as e:
            print(e)
            print("Not able to load given flow " + self.modelpath)
            exit(0)
        print("Sampling from flow " + self.modelpath)

        # Get unfolded sample
        with torch.no_grad():
          samples = flow_loaded.sample(1, context=val_trans_cond).view(val_trans_cond.shape[0], -1).cpu().numpy() # generate for a detector level condition one condition, this gives you the whole unfolded smaple!!


        ## inverse standardize
        retransformed_samples = trf.standardize_inverse(samples, mean_values[:,self.gen_index], std_values[:,self.gen_index])
        ## inverse logit
        retransformed_samples = trf.logit_inverse(retransformed_samples)
        ## inverse normalize
        retransformed_samples = trf.normalize_inverse(retransformed_samples, self.max_values[:,self.gen_index], self.min_values[:,self.gen_index])
        return retransformed_samples
