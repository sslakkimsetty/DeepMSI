import torch
import torch.nn as nn
from classification.build_model import build_model
from classification.model_opt import model_opt
from classification.build_dataloader import build_dataloader
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import logging


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('my_logger')

class classify(object):
    """ 
    classification class
    args:
      model_opt(obj): 

    """


    def __init__(self, model_opts, data, batch_size, optim, lr = 0.001, num_epochs = 50):
        super(classify, self).__init__()

        self.model_opts = model_opts
        self.data = data
        self.batch_size = batch_size
        self.optim = optim
        self.lr = lr
        self.num_epochs = num_epochs
        self.model = build_model(self.model_opts, gpu = False, checkpoint = None)
        self.dataloader = build_dataloader(csv_file = data, batch_size = self.batch_size)
    def train(self):
        model_params = self.model.parameters()
        optimizer = torch.optim.Adam(model_params, lr=self.lr)
        loss_func = torch.nn.CrossEntropyLoss()
        

        self.model.train()
        #### training
        for epoch in range(self.num_epochs):
          total = 0
          correct = 0
          losses = 0
          for i_batch, sample_batched in enumerate(self.dataloader):
            train_x = sample_batched['spectrum']
            train_y = sample_batched['t_label']
            optimizer.zero_grad()
            pred_y = self.model(train_x.float())
            loss = loss_func(pred_y, train_y.view(-1))
            loss.backward()
            optimizer.step()

            losses += loss.item()
            _, predicted = torch.max(pred_y.data, 1)
            total += train_y.size(0)
 
            correct += (predicted == train_y.view(-1)).sum().item()


          logger.info('Train Epoch: {} Loss: {:.6f} Accuracy: {:.6f}'.format(
                epoch + 1, losses/i_batch, correct/total))
    def val(self):
      with torch.no_grad():
          self.model.eval()
          loss_func = torch.nn.CrossEntropyLoss()
          losses = 0
          total = 0
          correct = 0
          pred_label = list()
          x_coord = list()
          y_coord = list()
          for i_batch, sample_batched in enumerate(self.dataloader):
              train_x = sample_batched['spectrum']
              train_y = sample_batched['t_label']
              pred_y = self.model(train_x.float())
              loss = loss_func(pred_y, train_y.view(-1))
              losses += loss.item()

              _, predicted = torch.max(pred_y.data, 1)
              total += train_y.size(0)
              correct += (predicted == train_y.view(-1)).sum().item()
              pred_label.extend(predicted)
              #print(sample_batched['x_coord'].view(-1).item())
              x_coord.extend(sample_batched['x_coord'].view(-1))
              y_coord.extend(sample_batched['y_coord'].view(-1))
      logger.info('Validation loss: {:.6f} Accuracy: {:.6f}'.format(
        losses/i_batch, correct/total))
      return pred_label, x_coord, y_coord
