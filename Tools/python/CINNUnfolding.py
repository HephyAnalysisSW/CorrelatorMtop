import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import json
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from CorrelatorMtop.Tools.H5Dataset import H5Dataset


class CINNUnfolding:
    """
    Class for cINN unfolding.
    Train the model with train() and unfold with predict()-
    """


    def __init__(self, train_data, val_data, rec_features, gen_features, save_path,
                 normalize=False, standardize=False,
                 gen_min=None, gen_max=None, gen_mean=None, gen_std=None,
                 rec_min=None, rec_max=None, rec_mean=None, rec_std=None):

        """
        Initialize CINNUnfolding with given attributes.

        Args:
            train_data (H5Dataset)    : Training data.
            val_data (H5Dataset)      : Validation data.
            rec_features (list of str): Detector level features.
            gen_features (list of str): Particle level features.
            save_path (str)           : Save path for the trained model.
            normalize (bool)          : Normalize the data?
            standardize (bool)        : Standardize the data?
            gen_min np.array)         : Can provide external values for normalization, otherwise calculated from train data.
            gen_max (np.array)        : Can provide external values for normalization, otherwise calculated from train data.
            gen_mean (np.array)       : Can provide external values for standardization, otherwise calculated from train data.
            gen_std (np.array)        : Can provide external values for standardization, otherwise calculated from train data.
            rec_min (np.array)        : Can provide external values for normalization, otherwise calculated from train data.
            rec_max (np.array)        : Can provide external values for normalization, otherwise calculated from train data.
            rec_mean (np.array)       : Can provide external values for standardization, otherwise calculated from train data.
            rec_std (np.array)        : Can provide external values for standardization, otherwise calculated from train data.

        """

        self.train_dataset = train_data
        self.val_dataset = val_data
        self.rec_features = rec_features
        self.gen_features = gen_features
        self.save_path = save_path

        self.batch_size = 1024
        self.learning_rate = 1e-4
        self.n_epochs = 25
        self.n_layers = 6
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss_train = []
        self.loss_val = []

        self.n_features = len(gen_features)
        self.n_context = len(rec_features)

        self.normalize = normalize
        self.standardize = standardize

        if normalize and standardize:
            raise ValueError("Choose either normalize or standardize, not both.")

        # If no external values are set here, calculate these later from the train data
        self.gen_min = np.array(gen_min) if gen_min is not None else None
        self.gen_max = np.array(gen_max) if gen_max is not None else None
        self.gen_mean = np.array(gen_mean) if gen_mean is not None else None
        self.gen_std = np.array(gen_std) if gen_std is not None else None

        self.rec_min = np.array(rec_min) if rec_min is not None else None
        self.rec_max = np.array(rec_max) if rec_max is not None else None
        self.rec_mean = np.array(rec_mean) if rec_mean is not None else None
        self.rec_std = np.array(rec_std) if rec_std is not None else None

        self.model = self.__build_flow()
        self.model.to(self.device)

        self.logger = None

    def __build_flow(self):
        """
        Build the normalizing flow
        """
        base_dist = StandardNormal(shape=[self.n_features])
        transforms = []
        for _ in range(self.n_layers):
            transforms.append(MaskedAffineAutoregressiveTransform(self.n_features, 64, context_features=self.n_context))
            transforms.append(ReversePermutation(features=self.n_features))
        return Flow(CompositeTransform(transforms), base_dist)

    def __compute_data_stats(self):
        """
        If you want to normalize/standardize but do not provide values for this,
        get them from the train data.
        """
        if self.normalize or self.standardize:
            all_gen = []
            all_rec = []
            for gen, rec in DataLoader(self.train_dataset, batch_size=1024):
                all_gen.append(gen)
                all_rec.append(rec)
            all_gen = torch.cat(all_gen, dim=0).numpy()
            all_rec = torch.cat(all_rec, dim=0).numpy()

            if self.normalize:
                if self.gen_min is None: self.gen_min = all_gen.min(axis=0)
                if self.gen_max is None: self.gen_max = all_gen.max(axis=0)
                if self.rec_min is None: self.rec_min = all_rec.min(axis=0)
                if self.rec_max is None: self.rec_max = all_rec.max(axis=0)

            if self.standardize:
                if self.gen_mean is None: self.gen_mean = all_gen.mean(axis=0)
                if self.gen_std is None: self.gen_std = all_gen.std(axis=0)
                if self.rec_mean is None: self.rec_mean = all_rec.mean(axis=0)
                if self.rec_std is None: self.rec_std = all_rec.std(axis=0)

    def _transform_gen(self, x):
        """
        normalize/standardize the particle level features
        """
        x = x.clone()
        if self.normalize:
            return (x - torch.tensor(self.gen_min)) / (torch.tensor(self.gen_max) - torch.tensor(self.gen_min))
        if self.standardize:
            return (x - torch.tensor(self.gen_mean)) / torch.tensor(self.gen_std)
        return x

    def _transform_rec(self, x):
        """
        normalize/standardize the detector level features
        """
        x = x.clone()
        if self.normalize:
            return (x - torch.tensor(self.rec_min)) / (torch.tensor(self.rec_max) - torch.tensor(self.rec_min))
        if self.standardize:
            return (x - torch.tensor(self.rec_mean)) / torch.tensor(self.rec_std)
        return x

    def invert_gen(self, x):
        """
        Invert normalize/standardize
        """
        if self.normalize:
            return x * (torch.tensor(self.gen_max) - torch.tensor(self.gen_min)) + torch.tensor(self.gen_min)
        if self.standardize:
            return x * torch.tensor(self.gen_std) + torch.tensor(self.gen_mean)
        return x

    def __save_loss(self):
        """
        Saves loss values in a npz file
        """
        loss_path = os.path.join(self.save_path, 'loss.npz')
        np.savez(loss_path,
            loss_train=np.array(self.loss_train),
            loss_val=np.array(self.loss_val)
            )
        if self.logger:
            self.logger.info(f"Saved loss: {loss_path}")

    def __save_norm_params(self):
        """
        Save parameters that were used for normalize/standardize in a json file.
        """
        norm_params = {
            'normalize': self.normalize,
            'standardize': self.standardize,
            'gen_min': self.gen_min.tolist() if self.gen_min is not None else None,
            'gen_max': self.gen_max.tolist() if self.gen_max is not None else None,
            'gen_mean': self.gen_mean.tolist() if self.gen_mean is not None else None,
            'gen_std': self.gen_std.tolist() if self.gen_std is not None else None,
            'rec_min': self.rec_min.tolist() if self.rec_min is not None else None,
            'rec_max': self.rec_max.tolist() if self.rec_max is not None else None,
            'rec_mean': self.rec_mean.tolist() if self.rec_mean is not None else None,
            'rec_std': self.rec_std.tolist() if self.rec_std is not None else None
        }
        norm_path = os.path.join(self.save_path, 'norm_params.json')
        with open(norm_path, 'w') as f:
            json.dump(norm_params, f)
        if self.logger:
            self.logger.info(f"Saved normalization parameters: {norm_path}")

    def __load_norm_params(self, model_path):
        """
        Get parameters that were used for normalize/standardize from a json file.
        """
        with open(os.path.join(os.path.dirname(model_path), 'norm_params.json'), 'r') as f:
            norm_params = json.load(f)

        self.normalize = norm_params['normalize']
        self.standardize = norm_params['standardize']
        self.gen_min = np.array(norm_params['gen_min']) if norm_params['gen_min'] is not None else None
        self.gen_max = np.array(norm_params['gen_max']) if norm_params['gen_max'] is not None else None
        self.gen_mean = np.array(norm_params['gen_mean']) if norm_params['gen_mean'] is not None else None
        self.gen_std = np.array(norm_params['gen_std']) if norm_params['gen_std'] is not None else None
        self.rec_min = np.array(norm_params['rec_min']) if norm_params['rec_min'] is not None else None
        self.rec_max = np.array(norm_params['rec_max']) if norm_params['rec_max'] is not None else None
        self.rec_mean = np.array(norm_params['rec_mean']) if norm_params['rec_mean'] is not None else None
        self.rec_std = np.array(norm_params['rec_std']) if norm_params['rec_std'] is not None else None

    def train(self):
        """
        Train the model
        """

        # Compute the parameters for normalize/standardize
        if self.logger:
            self.logger.info("Compute parameters for normalization...")
        self.__compute_data_stats()

        # Create the data loaders for the train/validation samples
        if self.logger:
            self.logger.info("Create DataLoaders...")

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # Set up the optimizer for the updating of the model
        if self.logger:
            self.logger.info("Create Optimizer...")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        os.makedirs(self.save_path, exist_ok=True)

        # Now we start the epoch loop
        if self.logger:
            self.logger.info("Start Epoch loop...")

        for epoch in range(1, self.n_epochs + 1):
            if self.logger:
                self.logger.info(f"Epoch {epoch}/{self.n_epochs} ...")

            self.model.train()
            train_loss = 0

            # load data in batches, transform, feed in model, make step and save loss
            for gen, rec in train_loader:
                gen = self._transform_gen(gen).to(self.device)
                rec = self._transform_rec(rec).to(self.device)

                optimizer.zero_grad()
                loss = -self.model.log_prob(inputs=gen, context=rec).mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.loss_train.append(train_loss)
            avg_train_loss = train_loss / len(train_loader)

            # Now save loss for validation data
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for gen, rec in val_loader:
                    gen = self._transform_gen(gen).to(self.device)
                    rec = self._transform_rec(rec).to(self.device)
                    loss = -self.model.log_prob(inputs=gen, context=rec).mean()
                    val_loss += loss.item()

            self.loss_val.append(val_loss)
            avg_val_loss = val_loss / len(val_loader)

            # Save model after each epoch
            print(f"Epoch {epoch}/{self.n_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            model_path = os.path.join(self.save_path, f"model_epoch{epoch}.pt")
            torch.save(self.model.state_dict(), model_path)

        # Save normalization stats and loss
        self.__save_norm_params()
        self.__save_loss()

        if self.logger:
            self.logger.info(f"Saved final model: {model_path}")

    def predict(self, rec_dataset, model_path, n_samples=1):
        # Load normalization/standardization parameters
        self.__load_norm_params(model_path)

        # Load the trained model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Collect reconstructed features from the dataset
        rec_data = []
        rec_loader = DataLoader(rec_dataset, batch_size=1024, shuffle=False)

        for _, rec in rec_loader:  # Each batch: (gen, rec), we only need rec
            rec_data.append(rec)

        # Stack all reco features into a tensor
        rec_array = torch.cat(rec_data, dim=0)
        rec_array = self._transform_rec(rec_array).to(torch.float32).to(self.device)

        # Generate n_samples of gen values for each reco event
        with torch.no_grad():
            gen_samples = self.model.sample(n_samples, context=rec_array)

        # Invert normalization to return gen predictions in original units
        gen_samples = self.invert_gen(gen_samples).cpu().numpy()

        return gen_samples
