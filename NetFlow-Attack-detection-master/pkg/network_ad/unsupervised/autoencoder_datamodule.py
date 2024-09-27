import sys
sys.path.append("../..")
import json
import multiprocessing
import os
import time
from typing import List

# import h5py
import h5pickle as h5py
import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from network_ad.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TRAIN_DATA_PATH, TEST_DATA_PATH, \
    NUMERICAL_STATS_PATH, CATEGORICAL_STATS_PATH, PREPROCESSED_DATA_PATH, AUTOENCODER_INPUT_DIMS
from network_ad.preprocessing.netflow_v2_preprocessing import train_test_split
import shutil


class AutoencoderDataset(Dataset):
    def __init__(self, hdf5_path, means, stds, mode: str):
        self.hdf5_path = hdf5_path
        self.means = means
        self.stds = stds
        self.h5file = h5py.File(self.hdf5_path, 'r')
        self.mode = mode

    def __len__(self):
        return self.h5file[self.mode].shape[0]

    def __getitem__(self, idx):
        """Return the features for a single sample."""
        features = self.h5file[self.mode][idx]
        return torch.tensor(features)





    def close(self):
        self.h5file.close()  # Ensure to close the HDF5 file when done


class AutoencoderDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, val_ratio=0.1, num_workers=0):
        super().__init__()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_path = TRAIN_DATA_PATH
        self.test_path = TEST_DATA_PATH
        self.numerical_stats_path = NUMERICAL_STATS_PATH
        self.categorical_stats_path = CATEGORICAL_STATS_PATH
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.means = None
        self.stds = None
        self.categorical_encoders = {}
        self.num_workers = num_workers
        self.binary_labels_by_mode = {}
        self.multiclass_labels_by_mode = {}

    def load_data(self, mode):
        if mode == 'train':
            df = pd.read_csv(self.train_path)
        elif mode == 'test':
            df = pd.read_csv(self.test_path)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Ensure that categorical features are read as strings
        for feature in CATEGORICAL_FEATURES:
            df[feature] = df[feature].astype(str)
        return df

    def get_binary_labels(self, mode)->List:
        """
        Only for visualization purposes.
        The training of the autoencoder is unsupervised.

        :param mode:
        :return:
        """
        if mode in self.binary_labels_by_mode:
            return self.binary_labels_by_mode[mode]

        if mode == 'train':
            df = pd.read_csv(self.train_path, usecols=['Label'])
            nb_train_samples = len(df) * (1-self.val_ratio)
            df = df[:int(nb_train_samples)]
            df['Label'] = df['Label'].apply(lambda x: 'Benign' if x == 0 else 'Malicious')
            self.binary_labels_by_mode[mode] = df['Label'].tolist()
            return df['Label'].tolist()

        elif mode == 'val':
            df = pd.read_csv(self.train_path, usecols=['Label'])
            nb_train_samples = len(df) * (1-self.val_ratio)
            df = df[int(nb_train_samples):]
            df['Label'] = df['Label'].apply(lambda x: 'Benign' if x == 0 else 'Malicious')
            self.binary_labels_by_mode[mode] = df['Label'].tolist()
            return df['Label'].tolist()
        elif mode == 'test':
            df = pd.read_csv(self.test_path, usecols=['Label'])
            df['Label'] = df['Label'].apply(lambda x: 'Benign' if x == 0 else 'Malicious')
            self.binary_labels_by_mode[mode] = df['Label'].tolist()
            return df['Label'].tolist()

    def get_multiclass_labels(self, mode)->List:
        """
        Only for visualization purposes.
        :param mode:
        :return:
        """
        if mode in self.multiclass_labels_by_mode:
            return self.multiclass_labels_by_mode[mode]

        if mode == 'train':
            df = pd.read_csv(self.train_path, usecols=['Attack'])
            nb_train_samples = len(df) * (1-self.val_ratio)
            df = df[:int(nb_train_samples)]
            self.multiclass_labels_by_mode[mode] = df['Attack'].tolist()
            return df['Attack'].tolist()

        elif mode == 'val':
            df = pd.read_csv(self.train_path, usecols=['Attack'])
            nb_train_samples = len(df) * (1-self.val_ratio)
            df = df[int(nb_train_samples):]
            self.multiclass_labels_by_mode[mode] = df['Attack'].tolist()
            return df['Attack'].tolist()

        elif mode == 'test':
            df = pd.read_csv(self.test_path, usecols=['Attack'])
            self.multiclass_labels_by_mode[mode] = df['Attack'].tolist()
            return df['Attack'].tolist()



    def setup(self, stage=None):
        """Load data and set up the encoders and statistics."""
        # Load statistics
        numerical_stats = pd.read_csv(self.numerical_stats_path, index_col=0)
        self.means = numerical_stats['mean'].values
        self.stds = numerical_stats['std'].values

        # Load categorical stats and fit one-hot encoders based on categorical_stats
        with open(self.categorical_stats_path, 'r') as f:
            categorical_stats = json.load(f)

        # Create one-hot encoders for each categorical feature
        for feature in CATEGORICAL_FEATURES:
            self.categorical_encoders[feature] = OneHotEncoder(sparse=False,
                                                               categories=[categorical_stats[feature]],
                                                               handle_unknown="ignore")
            self.categorical_encoders[feature].fit(np.array(categorical_stats[feature]).reshape(-1, 1))

        preprocessed_h5_path = PREPROCESSED_DATA_PATH / 'preprocessed_data.h5'
        temporary_h5_path = PREPROCESSED_DATA_PATH / f'temporary_data.h5'

        if temporary_h5_path.exists():
            temporary_h5_path.unlink()
        if not preprocessed_h5_path.exists():
            self.save_processed_dataset('train')
            self.save_processed_dataset('val')
            self.save_processed_dataset('test')

            # Move the temporary file to the final file
            shutil.copy(temporary_h5_path, preprocessed_h5_path)

        # Create dataset objects for training, validation, and test
        self.train_data = AutoencoderDataset(str(preprocessed_h5_path), self.means, self.stds, mode='train')
        self.val_data = AutoencoderDataset(str(preprocessed_h5_path), self.means, self.stds, mode='val')
        self.test_data = AutoencoderDataset(str(preprocessed_h5_path), self.means, self.stds, mode='test')

    def save_processed_dataset(self, mode):
        """Process and save dataset to HDF5."""

        if mode == 'train':
            df = self.load_data('train')
            nb_train_samples = len(df) * (1-self.val_ratio)
            dataframe = df[:int(nb_train_samples)]

        elif mode == 'val':
            df = self.load_data('train')
            nb_train_samples = len(df) *  (1-self.val_ratio)
            dataframe = df[int(nb_train_samples):]
        elif mode == 'test':
            df = self.load_data('test')
            dataframe = df

        # Create HDF5 file and write data
        temporary_h5_path = PREPROCESSED_DATA_PATH / f'temporary_data.h5'
        temporary_h5_sub_paths = [PREPROCESSED_DATA_PATH / f'temporary_data_chunk_{i}.h5' for i in
                                  range(multiprocessing.cpu_count())]

        def preprocesss_single_dataframe_chunk(dataframe_chunk, start_idx,
                                               mode,
                                               temporary_h5_sub_path):
            with h5py.File(temporary_h5_sub_path, 'w') as sub_f:
                sub_dataset = sub_f.create_dataset(mode, (len(dataframe_chunk), AUTOENCODER_INPUT_DIMS), dtype='f')
                # reset the index
                dataframe_chunk.reset_index(drop=True, inplace=True)
                for idx, row in tqdm(dataframe_chunk.iterrows(), total=len(dataframe_chunk),
                                     desc=f'Preprocessing {mode} . Chunk start index: {start_idx}. Chunk size: {len(dataframe_chunk)}'):
                    # One-hot encode categorical features using the pre-fitted encoders
                    categorical_data_list = []
                    for feature in CATEGORICAL_FEATURES:
                        encoded_feature = self.categorical_encoders[feature].transform([[row[feature]]])
                        categorical_data_list.append(encoded_feature)

                    # Normalize numerical features
                    numerical_data = row[NUMERICAL_FEATURES].values.astype(np.float32)
                    numerical_scaled = (numerical_data - self.means) / self.stds

                    # Concatenate numerical and one-hot encoded categorical features
                    categorical_encoded = np.hstack(categorical_data_list).squeeze()
                    features = np.hstack((numerical_scaled, categorical_encoded)).astype(np.float32)

                    # Write features to HDF5 dataset
                    sub_dataset[idx] = features


        for temporary_h5_sub_path in temporary_h5_sub_paths:
            if temporary_h5_sub_path.exists():
                temporary_h5_sub_path.unlink()

        with h5py.File(temporary_h5_path, 'a') as f:
            # Create dataset in HDF5 file
            dataset_name = mode
            n_chunks = multiprocessing.cpu_count()
            dataset_splits = np.array_split(dataframe, n_chunks)
            f.create_dataset(dataset_name, (len(dataframe), AUTOENCODER_INPUT_DIMS), dtype='f')
            start_idxs = [0]
            for i in range(1, n_chunks):
                start_idxs.append(start_idxs[i - 1] + len(dataset_splits[i - 1]))

            joblib.Parallel(n_jobs=n_chunks)(joblib.delayed(preprocesss_single_dataframe_chunk)(dataset_splits[i],
                                                                                                start_idxs[i], mode,
                                                                                                temporary_h5_sub_paths[
                                                                                                    i])
                                             for i in range(n_chunks))

            # Merge all sub files into one
            for i in range(n_chunks):
                with h5py.File(temporary_h5_sub_paths[i], 'r') as h5f_sub:
                    f[dataset_name][start_idxs[i]:start_idxs[i] + len(dataset_splits[i])] = h5f_sub[mode][:]
                    h5f_sub.close()
                # close the temporary file
                temporary_h5_sub_paths[i].unlink()

            f.close()

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False,
                          persistent_workers=True)

    def teardown(self, stage):
        if stage == 'test':
            self.test_data.close()
            self.train_data.close()
            self.val_data.close()



# Main function to test the DataModule
if __name__ == "__main__":
    # Define the batch size and validation ratio
    BATCH_SIZE = 64
    VAL_RATIO = 0.1

    # Initialize the DataModule
    data_module = AutoencoderDataModule(batch_size=BATCH_SIZE, val_ratio=VAL_RATIO)

    # Setup the data (loads, preprocesses, and splits the data)
    data_module.setup()

    # Test the train_dataloader
    print("Train DataLoader:")
    for batch in data_module.train_dataloader():
        print(f"Train Batch X: {batch.shape}")
        break  # Only show one batch

    # Test the val_dataloader
    print("Validation DataLoader:")
    for batch in data_module.val_dataloader():
        print(f"Validation Batch X: {batch.shape}")
        break  # Only show one batch

    # Test the test_dataloader
    print("Test DataLoader:")
    for batch in data_module.test_dataloader():
        print(f"Test Batch X: {batch.shape}")
        break  # Only show one batch
