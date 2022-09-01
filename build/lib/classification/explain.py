import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import lime
from lime import lime_tabular
import torch
from classification.build_dataloader import MSISpectrumDataset
from classification.build_dataloader import ToTensor
from classification import classify

def explain(model, train_data, test_data, instance_id = 1, num_features = 10):

  train = MSISpectrumDataset(data = train_data, transform=transforms.Compose([ToTensor()]))
  test = MSISpectrumDataset(data = test_data, transform=transforms.Compose([ToTensor()]))

  X_train = train.msi_dataset.iloc[:,4:]
  X_test = test.msi_dataset.iloc[:,4:]

  explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Non-tumor', 'Tumor'],
    mode='classification'
  )

  exp = explainer.explain_instance(
    data_row = X_test.iloc[instance_id],
    predict_fn = model.predict,
    num_features = num_features
  )
  
  #save figure
  fig = exp.as_pyplot_figure()
  fig.savefig('lime_report.jpg')
  
  keys = [x[0] for x in exp.as_list()]
  coef = [x[1] for x in exp.as_list()]
  out = np.array(list(zip(keys,coef)))


  return out
