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

from CorrelatorMtop.Tools.H5Dataset       import H5Dataset

class CINNUnfolding:
    def __init__(self, train_data, val_data, rec_features, gen_features, save_path):
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

        self.loss_training = []
        self.loss_validation = []

        self.n_features = len(gen_features)
        self.n_context = len(rec_features)

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

            print(f"Number of batches: {len(train_loader)}")
            i_batch = 0
            for gen, rec in train_loader:
                print(f"Batch {i_batch}")
                i_batch += 1

                gen = gen.to(self.device)
                rec = rec.to(self.device)

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
                    gen = gen.to(self.device)
                    rec = rec.to(self.device)
                    loss = -self.model.log_prob(inputs=gen, context=rec).mean()
                    val_loss += loss.item()

            self.loss_val.append(val_loss)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch}/{self.n_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            model_path = os.path.join(self.save_path, f"model_epoch{epoch}.pt")
            torch.save(self.model.state_dict(), model_path)
        if self.logger:
            logger.info(f"Saved final model: {model_path}")

    def predict(self, rec_data_file, model_path, n_samples=1):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        rec_data = []
        with h5py.File(rec_data_file, "r") as f:
            for i in range(len(f[self.rec_features[0]])):
                rec = np.stack([f[feat][i] for feat in self.rec_features], axis=0)
                rec_data.append(rec)
        rec_array = torch.tensor(rec_data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            samples = self.model.sample(n_samples, context=rec_array)
        return samples.cpu().numpy()
