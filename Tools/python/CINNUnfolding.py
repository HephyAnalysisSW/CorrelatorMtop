import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from CorrelatorMtop.Tools.H5Dataset import H5Dataset


class CINNUnfolding:
    def __init__(self, train_data, val_data, rec_features, gen_features, save_path,
                 normalize=False, standardize=False,
                 gen_min=None, gen_max=None, gen_mean=None, gen_std=None,
                 rec_min=None, rec_max=None, rec_mean=None, rec_std=None):

        self.train_dataset = train_data
        self.val_dataset = val_data
        self.rec_features = rec_features
        self.gen_features = gen_features
        self.save_path = save_path

        self.batch_size = 512
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

        # Stats (may be computed or provided)
        self.gen_min = np.array(gen_min) if gen_min is not None else None
        self.gen_max = np.array(gen_max) if gen_max is not None else None
        self.gen_mean = np.array(gen_mean) if gen_mean is not None else None
        self.gen_std = np.array(gen_std) if gen_std is not None else None

        self.rec_min = np.array(rec_min) if rec_min is not None else None
        self.rec_max = np.array(rec_max) if rec_max is not None else None
        self.rec_mean = np.array(rec_mean) if rec_mean is not None else None
        self.rec_std = np.array(rec_std) if rec_std is not None else None

        self.__compute_data_stats()

        self.model = self.__build_flow()
        self.model.to(self.device)

        self.logger = None

    def __build_flow(self):
        base_dist = StandardNormal(shape=[self.n_features])
        transforms = []
        for _ in range(self.n_layers):
            transforms.append(MaskedAffineAutoregressiveTransform(self.n_features, 64, context_features=self.n_context))
            transforms.append(ReversePermutation(features=self.n_features))
        return Flow(CompositeTransform(transforms), base_dist)

    def __compute_data_stats(self):
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
        x = x.clone()
        if self.normalize:
            return (x - torch.tensor(self.gen_min)) / (torch.tensor(self.gen_max) - torch.tensor(self.gen_min))
        if self.standardize:
            return (x - torch.tensor(self.gen_mean)) / torch.tensor(self.gen_std)
        return x

    def _transform_rec(self, x):
        x = x.clone()
        if self.normalize:
            return (x - torch.tensor(self.rec_min)) / (torch.tensor(self.rec_max) - torch.tensor(self.rec_min))
        if self.standardize:
            return (x - torch.tensor(self.rec_mean)) / torch.tensor(self.rec_std)
        return x

    def invert_gen(self, x):
        if self.normalize:
            return x * (torch.tensor(self.gen_max) - torch.tensor(self.gen_min)) + torch.tensor(self.gen_min)
        if self.standardize:
            return x * torch.tensor(self.gen_std) + torch.tensor(self.gen_mean)
        return x

    def train(self):
        if self.logger:
            self.logger.info("Create DataLoaders...")

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        if self.logger:
            self.logger.info("Create Optimizer...")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        os.makedirs(self.save_path, exist_ok=True)

        if self.logger:
            self.logger.info("Start Epoch loop...")

        for epoch in range(1, self.n_epochs + 1):
            if self.logger:
                self.logger.info(f"Epoch {epoch}/{self.n_epochs} ...")

            self.model.train()
            train_loss = 0

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

            print(f"Epoch {epoch}/{self.n_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            model_path = os.path.join(self.save_path, f"model_epoch{epoch}.pt")
            torch.save(self.model.state_dict(), model_path)

        if self.logger:
            self.logger.info(f"Saved final model: {model_path}")

    def predict(self, rec_data_file, model_path, n_samples=1):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        rec_data = []
        with h5py.File(rec_data_file, "r") as f:
            for i in range(len(f[self.rec_features[0]])):
                rec = np.stack([f[feat][i] for feat in self.rec_features], axis=0)
                rec_data.append(rec)

        rec_array = torch.tensor(rec_data, dtype=torch.float32)
        rec_array = self._transform_rec(rec_array).to(self.device)

        with torch.no_grad():
            samples = self.model.sample(n_samples, context=rec_array)

        return samples.cpu().numpy()
