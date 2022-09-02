import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class MSISpectrumDataset(Dataset):
    """collections of spectra in a Mass spectrometry imaging dataset """

    def __init__(self, data, sampler = 'RandomSampler', transform=None):
        """
        Args:
            data: Dataframe or csv_file (string), Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if type(data) == str:
          self.msi_dataset = pd.read_csv(data, index_col = 0)
          self.msi_dataset = self.msi_dataset.reset_index(drop=True)
        else:
          self.msi_dataset = data
        self.msi_dataset['label'] = self.msi_dataset['label'].astype(int)
        self.msi_dataset['sample_id'] = self.msi_dataset['sample_id'].astype(object)
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

def build_dataloader(data, batch_size = 32, random = True):
  transformed_dataset = MSISpectrumDataset(data = data,
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
  dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle = random, num_workers=0)
  return dataloader


def build_dataloader_mi(data, batch_size = 32, random = True, pred = None, ratio = 0.1):
  transformed_dataset = MSISpectrumDataset(data = data,
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
  pred['label'] = pred['label'].astype(dtype ='int64')
  for sample_idx in transformed_dataset.msi_dataset['sample_id'].unique():
    pos = transformed_dataset.msi_dataset.label[transformed_dataset.msi_dataset['sample_id']==sample_idx].sum()
    if pos:
      ids = np.array(transformed_dataset.msi_dataset['sample_id']== sample_idx)
      transformed_dataset.msi_dataset.label[ids] = pred.label[ids]

      if pred.label[ids].sum() != 0:

        k = int(ratio*ids.sum())

        id_pos = pred.prob[ids].nlargest(k).index.values

        transformed_dataset.msi_dataset.label[id_pos] = int(1)


    
  dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle = random, num_workers=0)
  return dataloader
