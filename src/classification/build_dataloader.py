import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class MSISpectrumDataset(Dataset):
    """collections of spectra in a Mass spectrometry imaging dataset """

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.msi_dataset = pd.read_csv(csv_file, index_col = 0)
        self.csv_file = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.msi_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # tissue level annotations
        t_label = self.msi_dataset.iloc[idx, 0].reshape(1)

        # sample id
        sample_id = self.msi_dataset.iloc[idx, 1]

        # x-y coordinates of spectra
        x_coord = self.msi_dataset.iloc[idx, 2].reshape(1)
        y_coord = self.msi_dataset.iloc[idx, 3].reshape(1)

        # spectrum data
        spectrum = self.msi_dataset.iloc[idx, 4:]
        spectrum = np.array([spectrum])
        spectrum = spectrum.astype('float').reshape(1,-1)

        sample = {'spectrum': spectrum, 't_label': t_label, 'sample_id': sample_id,
                  'x_coord': x_coord, 'y_coord': y_coord}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        spectrum, t_label = sample['spectrum'], sample['t_label']
        x_coord, y_coord = sample['x_coord'], sample['y_coord']

        return {'spectrum': torch.from_numpy(spectrum),
                't_label': torch.from_numpy(t_label),
                'x_coord': torch.from_numpy(x_coord),
                'y_coord': torch.from_numpy(y_coord)
                 }

def build_dataloader(csv_file = 'bladder_train.csv', batch_size = 32):
  transformed_dataset = MSISpectrumDataset(csv_file = csv_file,
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
  dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
  return dataloader